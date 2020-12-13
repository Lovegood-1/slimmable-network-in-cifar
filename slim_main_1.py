'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models




model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn_slim',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def forward_loss(model, criterion, input, target, meter, soft_target=None,soft_criterion=None, return_soft_target=False, return_acc=False):
    """forward model and return loss"""
    output = model(input)
    if soft_target is not None:
        loss = torch.mean(soft_criterion(output, soft_target))
    else:
        loss = torch.mean(criterion(output, target))
    # topk
    # _, pred = output.topk(max(FLAGS.topk))
    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct_k = []
    # for k in FLAGS.topk:
    #     correct_k.append(correct[:k].float().sum(0))
    # tensor = torch.cat([loss.view(1)] + correct_k, dim=0)
    # # allreduce
    # tensor = dist_all_reduce_tensor(tensor)
    # # cache to meter
    # tensor = tensor.cpu().detach().numpy()
    # bs = (tensor.size-1)//2
    # for i, k in enumerate(FLAGS.topk):
    #     error_list = list(1.-tensor[1+i*bs:1+(i+1)*bs])
    #     if return_acc and k == 1:
    #         top1_error = sum(error_list) / len(error_list)
    #         return loss, top1_error
    #     if meter is not None:
    #         meter['top{}_error'.format(k)].cache_list(error_list)
    # if meter is not None:
    #     meter['loss'].cache(tensor[0])
    # if return_soft_target:
    #     return loss, torch.nn.functional.softmax(output, dim=1)
    return loss, output



def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc
    # batch_time = AverageMeter()
    batch_time_dict = [AverageMeter() for i in range(4)]

    # data_time = AverageMeter()
    data_time_dict = [AverageMeter() for i in range(4)]

    # losses = AverageMeter()
    losses_dict = [AverageMeter() for i in range(4)]

    # top1 = AverageMeter()
    top1_dict = [AverageMeter() for i in range(4)]

    # top5 = AverageMeter()
    top5_dict = [AverageMeter() for i in range(4)]


    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    lenth_ = len(testloader)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        for width_mult in sorted(mul_list, reverse=True):
            model.apply(lambda m: setattr(m,'width_mult',width_mult))

            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            idx = mul_list.index(width_mult)
            losses = losses_dict[idx]
            top1 = top1_dict[idx]
            top5 = top5_dict[idx]
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        # compute output
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # plot progress
            if batch_idx == lenth_ -1:
                print('({batch}/{size}) Mul: {Mul:.3f}s |  Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx + 1,
                            size=len(testloader),
                            Mul=width_mult,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            ))

    return
mul_list = [0.25,0.5,0.75,1]
def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    #batch_time = AverageMeter()
    batch_time_dict = [AverageMeter() for i in range(4)]

    #data_time = AverageMeter()
    data_time_dict = [AverageMeter() for i in range(4)]

    #losses = AverageMeter()
    losses_dict = [AverageMeter() for i in range(4)]

    #top1 = AverageMeter()
    top1_dict = [AverageMeter() for i in range(4)]

    #top5 = AverageMeter()
    top5_dict = [AverageMeter() for i in range(4)]
    end = time.time()
    lr = optimizer.param_groups[0]['lr']
    leng_ = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # measure data loading time
        # data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            pass
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        for width_mult in sorted(mul_list, reverse=True):
            model.apply(lambda m: setattr(m,'width_mult',width_mult))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            idx = mul_list.index(width_mult)
            losses = losses_dict[idx]
            top1 = top1_dict[idx]
            top5 = top5_dict[idx]
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            # optimizer.zero_grad()


            # measure elapsed time
            # batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if batch_idx % 100 == 0 or batch_idx == leng_ -1:
                print('({batch}/{size}) mul: {mul:.3f}s |  Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}| lr: {lr:.4f}'.format(
                            batch=batch_idx + 1,
                            size=len(trainloader),
                            mul= width_mult,

                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            lr = lr
                            ))

    return

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch


    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='../data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model ")
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )

    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    # from  vgg_git import vgg19_bn
    # model = vgg19_bn()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test(testloader, model, criterion, epoch, use_cuda)

        # append logger file

        # save model





if __name__ == '__main__':
    main()
