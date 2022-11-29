import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import copy

import time

from models.resnet32_cifar10_grasp import resnet32
from models.vgg_grasp import vgg19, vgg16
from models.resnet20_cifar import resnet20
from models.resnet18_cifar import resnet18

from torch.optim.lr_scheduler import _LRScheduler

from testers import *

import numpy as np
import numpy.random as npr

import random
from prune_utils import *

# CL dataset and buffer library
from datasets import get_dataset
from utils.buffer import Buffer


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
# parser.add_argument('--dataset', type=str, default="cifar10",
#                     help='[cifar10, cifar100]')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.3, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--rho', type=float, default = 0.0001,
                    help ="Just for initialization")
parser.add_argument('--pretrain-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pretrain')
parser.add_argument('--pruning-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for pruning')
parser.add_argument('--remark', type=str, default=None,
                    help='optimizer used (default: adam)')
parser.add_argument('--save-model', type=str, default='model/',
                    help='optimizer used (default: adam)')
parser.add_argument('--sparsity-type', type=str, default='random-pattern',
                    help="define sparsity_type: [irregular,column,filter,pattern]")
parser.add_argument('--config-file', type=str, default='config_vgg16',
                    help="config file name")


# ------- argments for CL setup ----------
parser.add_argument('--use_cl_mask', action='store_true', default=False, help='use CL mask or not')
parser.add_argument('--buffer-size', type=int, default=500, metavar='N',
                    help='buffer size for class incremental training (default: 100)')
parser.add_argument('--buffer_weight', type=float, default=1.0, help="weight of ce loss of buffered samples")
parser.add_argument('--buffer_weight_beta', type=float, default=1.0, help="weight of ce loss of buffered samples in DERPP")
parser.add_argument('--dataset', type=str, default="seq-cifar10",
                    help='[seq-cifar10, seq-cifar100]')
parser.add_argument('--validation', action='store_true', default=False,
                    help='CL validation T of F')    
parser.add_argument('--test_epoch_interval', type=int, default=1, metavar='how often we do test',
                    help='buffer size for class incremental training (default: 100)')
parser.add_argument('--evaluate_mode', action='store_true', default=False, help='if we want to evaluate the checkpoints')
parser.add_argument("--eval_checkpoint", default=None, type=str, metavar="PATH", help="path to evalute checkpoint (default: none)")
parser.add_argument('--gradient_efficient', action='store_true', default=False,
                    help='add gradient efficiency')   
parser.add_argument('--gradient_efficient_mix', action='store_true', default=False,
                    help='add gradient efficiency (mix method)')     
parser.add_argument('--gradient_remove', type=float, default=0.1, help="extra removal for gradient efficiency")
parser.add_argument('--gradient_sparse', type=float, default=0.75,
                    help="total gradient_sparse for training")
parser.add_argument('--sample_frequency', type=int, default=30, help="sample frequency for gradient mask")
parser.add_argument('--replay_method', type=str, default='er', help='replay method to use')


parser.add_argument('--patternNum', type=int, default=8, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--rand-seed', action='store_true', default=False,
                    help='use random seed')
parser.add_argument("--log-filename", default=None, type=str, help='log filename, will override self naming')
parser.add_argument("--resume", default=None, type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument('--save-mask-model', action='store_true', default=False, help='save a sparse model indicating pruning mask')
parser.add_argument('--mask-sparsity', type=str, default=None, help='dir and file name for mask models')

parser.add_argument('--output-dir', required=True, help='directory where to save results')
parser.add_argument('--output-name', type=str, required=True)
parser.add_argument('--remove-data-epoch', type=int, default=200,
                    help='the epoch to remove partial training dataset')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--remove-n', type=int, default=0,
                    help='number of sorted examples to remove from training')
parser.add_argument('--keep-lowest-n', type=int, default=0,
                    help='number of sorted examples to keep that have the lowest score, equivalent to start index of removal, if a negative number given, remove random draw of examples')
parser.add_argument('--sorting-file', type=str, default=None, help='input file name for sorted pkl file')
parser.add_argument('--input-dir', type=str, default=".", help='input dir for sorted pkl file')

prune_parse_arguments(parser)
args = parser.parse_args()

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.rand_seed:
    seed = random.randint(1, 999)
    print("Using random seed:", seed)
else:
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    print("Using manual seed:", seed)

if not os.path.exists(args.save_model):
    os.makedirs(args.save_model)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def train(model, trainset, criterion, scheduler, optimizer, epoch, t, buffer, dataset,
    example_stats_train, train_indx, maskretrain, masks, cl_mask=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_loss = 0.
    correct = 0.
    total = 0.
    # switch to train mode
    model.train()

    # Get permutation to shuffle trainset
    trainset_permutation_inds = npr.permutation(
        np.arange(len(trainset.targets)))   #numpy random permutation
    batch_size = args.batch_size
    end = time.time()
    for batch_idx, batch_start_ind in enumerate(
            range(0, len(trainset.targets), batch_size)):
        data_time.update(time.time() - end)

        # prune_update_learning_rate(optimizer, epoch, args)

        # Get trainset indices for batch
        batch_inds = trainset_permutation_inds[batch_start_ind:
                                               batch_start_ind + batch_size]
        if len(batch_inds) < args.batch_size:
            continue
        # Get batch inputs and targets, transform them appropriately
        transformed_trainset = []
        not_transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
            not_transformed_trainset.append(trainset.__getitem__(ind)[2])
        inputs = torch.stack(transformed_trainset)
        not_transformed_inputs = torch.stack(not_transformed_trainset)
        targets = torch.LongTensor(np.array(trainset.targets)[batch_inds].tolist())

        # Map to available device
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if args.mixup:
            inputs, target_a, target_b, lam = mixup_data(inputs, targets, args.alpha)


        # Forward propagation, compute loss, get predictions
        # add buffer here
        if (not buffer is None) and (not buffer.is_empty()) and t > 0:
            if args.replay_method == "er":
                buf_inputs, buf_labels = buffer.get_data(
                    args.batch_size, transform=dataset.get_transform())
                if not args.merge_batch or (t == 0):
                    # compute output
                    outputs = model(inputs)
                    # add CL per task mask
                    if cl_mask is not None:
                        mask_add_on = torch.zeros_like(outputs)
                        mask_add_on[:, cl_mask] = float('-inf')
                        cl_masked_output = outputs + mask_add_on
                        ce_loss = criterion(cl_masked_output, targets)
                    else:
                        ce_loss = criterion(outputs, targets)   
                    # do an additional forward
                        # print("Buffer training!")
                    buf_output = model(buf_inputs)
                    buf_ce_loss = criterion(buf_output, buf_labels)
                    # ce_loss = ce_loss.mean()
                    # buf_ce_loss = buf_ce_loss.mean()
                    ce_loss += args.buffer_weight * buf_ce_loss
                else:
                    assert buffer is not None, "merge batch is not available when buffer is None!"
                    cat_inputs = torch.cat([inputs, buf_inputs], dim=0)
                    cat_targets = torch.cat([targets, buf_labels])
                    # compute output
                    cat_outputs = model(cat_inputs)
                    # make sure only count non-buffer data
                    outputs = cat_outputs[:args.batch_size]
                    # add CL per task mask
                    if cl_mask is not None:
                        mask_add_on = torch.zeros_like(cat_outputs)
                        # only add mask for the first half of batch
                        mask_add_on[:args.batch_size, cl_mask] = float('-inf')
                        cl_masked_output = cat_outputs + mask_add_on
                        ce_loss = criterion(cl_masked_output, cat_targets)
                    else:
                        ce_loss = criterion(cat_outputs, cat_targets)
                

            else: # if using der or derpp
                # compute output
                outputs = model(inputs)
                # add CL per task mask
                if cl_mask is not None:
                    mask_add_on = torch.zeros_like(outputs)
                    mask_add_on[:, cl_mask] = float('-inf')
                    cl_masked_output = outputs + mask_add_on
                    ce_loss = criterion(cl_masked_output, targets)
                else:
                    ce_loss = criterion(outputs, targets)
                # print(inputs.shape)

                if args.replay_method == "der":
                    buf_inputs, buf_logits = buffer.get_data(
                        args.batch_size, transform=dataset.get_transform())
                    buf_output = model(buf_inputs)
                    buf_mse_loss = F.mse_loss(buf_output, buf_logits, reduction="none")
                    buf_mse_loss = torch.mean(buf_mse_loss, axis=-1)
                    # ce_loss = ce_loss.mean()
                    # buf_mse_loss = buf_mse_loss.mean()
                    ce_loss += args.buffer_weight * buf_mse_loss

                elif args.replay_method == "derpp":
                    buf_inputs, _, buf_logits = buffer.get_data(
                        args.batch_size, transform=dataset.get_transform())
                    buf_output = model(buf_inputs)
                    # print(buf_inputs.shape)
                    buf_mse_loss = F.mse_loss(buf_output, buf_logits, reduction="none")
                    buf_mse_loss = torch.mean(buf_mse_loss, axis=-1)
                    # print(ce_loss.shape, buf_mse_loss.shape)
                    
                    # ce_loss = ce_loss.mean()
                    # buf_mse_loss = buf_mse_loss.mean()
                    ce_loss += args.buffer_weight * buf_mse_loss

                    buf_inputs, buf_labels, _ = buffer.get_data(
                        args.batch_size, transform=dataset.get_transform())
                    # print(buf_inputs.shape)
                    buf_output = model(buf_inputs)
                    buf_ce_loss = criterion(buf_output, buf_labels)
                    # print(ce_loss.shape, buf_ce_loss.shape)
                    # exit(0)
                    # buf_ce_loss = buf_ce_loss.mean()
                    ce_loss += args.buffer_weight_beta * buf_ce_loss
                
        else: # no replay
            # compute output
            outputs = model(inputs)
            # add CL per task mask
            if cl_mask is not None:
                mask_add_on = torch.zeros_like(outputs)
                mask_add_on[:, cl_mask] = float('-inf')
                cl_masked_output = outputs + mask_add_on
                ce_loss = criterion(cl_masked_output, targets)
            else:
                ce_loss = criterion(outputs, targets)
        loss = ce_loss


        # loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)

        # Update statistics and loss
        acc = predicted == targets

        for j, index in enumerate(batch_inds):

            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = train_indx[index]

            # Compute missclassification margin
            output_correct_class = outputs.data[j, targets[j].item()]
            sorted_output, _ = torch.sort(outputs.data[j, :])
            if acc[j]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[-2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = example_stats_train.get(index_in_original_dataset,
                                            [[], [], []])

            index_stats[0].append(loss[j].item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            example_stats_train[index_in_original_dataset] = index_stats

        # Update loss, backward propagate, update optimizer
        #print('inside len(example_stats_train)',len(example_stats_train))

        # losses.update(loss.item(), inputs.size(0))


        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        if args.gradient_efficient:
            prune_apply_masks_on_grads_efficient()
        elif args.gradient_efficient_mix:
            if batch_idx % args.sample_frequency == 0:
                prune_apply_masks_on_grads_mix()
            else:
                prune_apply_masks_on_grads_efficient()
        else:
            prune_apply_masks_on_grads()
        optimizer.step()

        optimizer.zero_grad()
        prune_apply_masks()

        batch_time.update(time.time() - end)
        end = time.time()

        # Add training accuracy to dict
        index_stats = example_stats_train.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats_train['train'] = index_stats

        # add data to buffer at the very end of the training iteration
        # Note that the datas are already transformed
        if args.replay_method == 'er':
            buffer.add_data(examples=not_transformed_inputs, labels=targets)
        elif args.replay_method == 'der':
            buffer.add_data(examples=not_transformed_inputs, logits=outputs.data)
        elif args.replay_method == 'derpp':
            buffer.add_data(examples=not_transformed_inputs, labels=targets, logits=outputs.data)

        if batch_idx % 10 == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR: {3:.5f}\t'
                  'Loss {4:.4f}\t'
                  'Acc@1 {5:.3f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  .format(
                    epoch, batch_idx, (len(trainset) // batch_size) + 1,
                    current_lr,
                    loss.item(), 100. * correct.item() / total,
                    batch_time=batch_time,
                    data_time=data_time
            ))


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mask_classes(outputs, dataset, k):
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')
            

def test(model, dataset):
    model.eval()
    acc_list = np.zeros((dataset.N_TASKS, ))
    til_acc_list = np.zeros((dataset.N_TASKS, ))
    with torch.no_grad():
        for task, test_loader in enumerate(dataset.test_loaders):
            test_loss = 0
            correct = 0
            til_correct = 0
            for data in test_loader:
                img, target = data
                # print(f"\tTest classes"+str(np.unique(target)))
                if args.cuda:
                    img, target = img.cuda(), target.cuda()
                img, target = Variable(img, volatile=True), Variable(target)
                output = model(img)
                criterion = nn.CrossEntropyLoss()
                test_loss = criterion(output, target)
                # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # pred for task incremental
                mask_classes(output, dataset, task)
                til_pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                til_correct += til_pred.eq(target.data.view_as(til_pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            acc = float(100. * correct) / float(len(test_loader.dataset))
            til_acc = float(100. * til_correct) / float(len(test_loader.dataset))
            acc_list[task] = acc
            til_acc_list[task] = til_acc
            print(f"Task {task}, Average loss {test_loss:.4f}, Class inc Accuracy {acc:.3f}, Task inc Accuracy {til_acc:.3f}")

    return acc_list, til_acc_list


def evaluate(model, dataset, last=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs = np.zeros((dataset.N_TASKS, ))
    accs_mask_classes = np.zeros((dataset.N_TASKS, ))
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                # if 'class-il' not in model.COMPATIBILITY:
                #     outputs = model(inputs, k)
                # else:
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                # if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        # accs.append(correct / total * 100)
        accs[k] = correct / total * 100
        # accs_mask_classes.append(correct_mask_classes / total * 100)
        accs_mask_classes[k] = correct_mask_classes / total * 100

    return accs, accs_mask_classes


def compute_forgetting_statistics(diag_stats, npresentations):

    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}
    print('len(diag_stats.items())',len(diag_stats.items()))

    for example_id, example_stats in diag_stats.items():

        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):

            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[1][:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example, 
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the misclassication margin for each presentation of the example
            margins_per_presentation = np.array(
                example_stats[2][:npresentations])

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned


def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):

    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all[0].keys():

        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example
        for i in range(len(unlearned_per_presentation_all)):

            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)

    num_unforget = len(np.where(np.array(example_stats) == 0)[0])
    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    return np.array(example_original_order)[np.argsort(
        example_stats)], np.sort(example_stats), num_unforget


def check_filename(fname, args_list):

    # If no arguments are specified to filter by, pass filename
    if args_list is None:
        return 1

    for arg_ind in np.arange(0, len(args_list), 2):
        arg = args_list[arg_ind]
        arg_value = args_list[arg_ind + 1]

        # Check if filename matches the current arg and arg value
        if arg + '_' + arg_value + '__' not in fname:
            print('skipping file: ' + fname)
            return 0

    return 1


# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def main():
    if args.cuda:
        if args.arch == "vgg":
            if args.depth == 19:
                model = vgg19(dataset=args.dataset)
            elif args.depth == 16:
                model = vgg16(dataset=args.dataset)
            else:
                sys.exit("vgg doesn't have those depth!")
        elif args.arch == "resnet":
            if args.depth == 18:
                model = resnet18(dataset=args.dataset)
            elif args.depth == 20:
                model = resnet20(dataset=args.dataset)
            elif args.depth == 32:
                model = resnet32(depth=32, dataset=args.dataset)
            else:
                sys.exit("resnet doesn't implement those depth!")
        else:
            sys.exit("wrong arch!")

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduce=False)


    # ----------- load checkpoint ---------------------
    model_state = None
    current_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                model_state = checkpoint['state_dict']
                current_epoch = checkpoint['current_epoch']
            else:
                model_state = checkpoint

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            # time.sleep(1)
            model_state = None
        time.sleep(2)

    if not model_state is None:
        model.load_state_dict(model_state)

    log_filename = args.log_filename
    print(log_filename)

    log_filename_dir_str = log_filename.split('/')
    log_filename_dir = "/".join(log_filename_dir_str[:-1])
    if not os.path.exists(log_filename_dir):
        os.system('mkdir -p ' + log_filename_dir)
        print("New folder {} created...".format(log_filename_dir))

    with open(log_filename, 'a') as f:
        for arg in sorted(vars(args)):
            f.write("{}:".format(arg))
            f.write("{}".format(getattr(args, arg)))
            f.write("\n")

    # ------------- pre training ---------------------
    print("==============pre training=================")

    prune_init(args, model)
    prune_apply_masks()  # if wanted to make sure the mask is applied in retrain
    prune_print_sparsity(model)

    _, total_sparsity = test_sparsity(model, column=False, channel=False, filter=False, kernel=False)


    # CL buffer and dataset setup
    dataset = get_dataset(args)
    print("*"*10 + f"Inspecting {args.dataset}" + "*"*10)
    print("*"*10 + "Initializing buffer" + "*"*10)

    if args.buffer_size > 0:
        buffer = Buffer(args.buffer_size, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        buffer = None

    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))  

    # Initialize dictionary to save statistics for every example presentation
    # example_stats_train = {}  # change name because fogetting function also have example_stats

    for t in range(dataset.N_TASKS):
        # do it per task
        example_stats_train = {}  

        optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

        optimizer = None
        if (args.optmzr == 'sgd'):
            # optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr) # CL no momentum and wd
        elif (args.optmzr == 'adam'):
            optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

        scheduler = None

        # initialize training dataset and full dataset here
        _, _, train_dataset, _ = dataset.get_data_loaders(return_dataset=True)
        full_dataset = copy.deepcopy(train_dataset)

        if args.sorting_file == None:
            train_indx = np.array(range(len(full_dataset.targets)))
        else:
            try:
                with open(
                        os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                        'rb') as fin:
                    ordered_indx = pickle.load(fin)['indices']
            except IOError:
                with open(os.path.join(args.input_dir, args.sorting_file),
                          'rb') as fin:
                    ordered_indx = pickle.load(fin)['indices']

            # Get the indices to remove from training
            elements_to_remove = np.array(ordered_indx)[-1:-1 + args.remove_n]
            print('elements_to_remove', len(elements_to_remove))

            # Remove the corresponding elements
            train_indx = np.setdiff1d(range(len(train_dataset.targets)), elements_to_remove)
            print('train_indx', len(train_indx))

        # Reassign train data and labels and save the removed data
        train_dataset.data = full_dataset.data[train_indx, :, :, :]
        print(train_dataset.data.shape)  # (35000, 32, 32, 3)

        train_dataset.targets = np.array(full_dataset.targets)[train_indx].tolist()
        print('len(train_dataset.targets)', len(train_dataset.targets))


        if args.use_cl_mask:
            cur_classes = np.arange(t*dataset.N_CLASSES_PER_TASK, (t+1)*dataset.N_CLASSES_PER_TASK)
            cl_mask = np.setdiff1d(np.arange(dataset.TOTAL_CLASSES), cur_classes)
        else:
            cl_mask = None

        for epoch in range(int(args.epochs/dataset.N_TASKS)):
            prune_update(epoch)

            #########remove data at 25 epoch, update dataset ######
            if epoch > 0 and epoch % args.sp_mask_update_freq == 0 and epoch <= args.remove_data_epoch:
                if args.sorting_file == None:
                    print('epoch', epoch)

                    unlearned_per_presentation_all, first_learned_all = [], []

                    _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(example_stats_train, int(args.epochs/dataset.N_TASKS))
                    print('unlearned_per_presentation', len(unlearned_per_presentation))
                    print('first_learned', len(first_learned))

                    unlearned_per_presentation_all.append(unlearned_per_presentation)
                    first_learned_all.append(first_learned)

                    print('unlearned_per_presentation_all', len(unlearned_per_presentation_all))
                    print('first_learned_all', len(first_learned_all))

                    # print('epoch before sort ordered_examples len',len(ordered_examples))

                    # Sort examples by forgetting counts in ascending order, over one or more training runs
                    ordered_examples, ordered_values, num_unforget = sort_examples_by_forgetting(unlearned_per_presentation_all, first_learned_all, int(args.epochs/dataset.N_TASKS))

                    # Save sorted output
                    if args.output_name.endswith('.pkl'):
                        with open(os.path.join(args.output_dir, args.output_name + "_task_"+ str(t) + "_unforget_"+str(num_unforget)),
                                'wb') as fout:
                            pickle.dump({
                                'indices': ordered_examples,
                                'forgetting counts': ordered_values
                            }, fout)
                    else:
                        with open(
                                os.path.join(args.output_dir, args.output_name + "_task_"+ str(t) + "_unforget_"+str(num_unforget) + '.pkl'),
                                'wb') as fout:
                            pickle.dump({
                                'indices': ordered_examples,
                                'forgetting counts': ordered_values
                            }, fout)

                    # Get the indices to remove from training
                    print('epoch before ordered_examples len', len(ordered_examples))
                    print('epoch before len(train_dataset.targets)', len(train_dataset.targets))
                    elements_to_remove = np.array(
                        ordered_examples)[args.keep_lowest_n:args.keep_lowest_n + ( int(args.remove_n/( int(args.remove_data_epoch)/args.sp_mask_update_freq ) ) )]
                    # Remove the corresponding elements
                    print('elements_to_remove', len(elements_to_remove))

                    train_indx = np.setdiff1d(
                        # range(len(train_dataset.targets)), elements_to_remove)
                        train_indx, elements_to_remove)
                    print('removed train_indx', len(train_indx))

                    # Reassign train data and labels
                    train_dataset.data = full_dataset.data[train_indx, :, :, :]
                    train_dataset.targets = np.array(
                        full_dataset.targets)[train_indx].tolist()

                    print('shape', train_dataset.data.shape)
                    print('len(train_dataset.targets)', len(train_dataset.targets))

                    # print('epoch after random ordered_examples len', len(ordered_examples))
                    #####empty example_stats_train!!! Because in original, forget process come before the whole training process
                    example_stats_train = {}

                ##########

            print('Training on ' + str(len(train_dataset.targets)) + ' examples')

            train(model, train_dataset, criterion, scheduler, optimizer, epoch, t, buffer, dataset,
                example_stats_train, train_indx, maskretrain=False, masks={}, cl_mask=cl_mask)

            prune_print_sparsity(model)
            if args.gradient_efficient or args.gradient_efficient_mix:
                show_mask_sparsity()

            if epoch % args.test_epoch_interval == 0 or epoch == (int(args.epochs/dataset.N_TASKS)-1):
                acc_list, til_acc_list = evaluate(model, dataset)
                prec1 = sum(acc_list) / (t+1)
                til_prec1 = sum(til_acc_list) / (t+1)
                acc_matrix[t] = acc_list
                forgetting = np.mean((np.max(acc_matrix, axis=0) - acc_list)[:t]) if t > 0 else 0.0
                learning_acc = np.mean(np.diag(acc_matrix)[:t+1])

                
                lr = optimizer.param_groups[0]['lr']
                log_line = 'Training on ' + str(len(train_dataset.targets)) + ' examples\n'
                log_line += f"Task: {t}, Epoch:{epoch}, Average Acc:[{prec1:.3f}], , Task Inc Acc:[{til_prec1:.3f}], Learning Acc:[{learning_acc:.3f}], Forgetting:[{forgetting:.3f}], LR:{lr}\n"
                log_line += "\t"
                for i in range(t+1):
                    log_line += f"Acc@T{i}: {acc_list[i]:.3f}\t"
                log_line += "\n"
                log_line += "\t"
                for i in range(t+1):
                    log_line += f"Til-Acc@T{i}: {til_acc_list[i]:.3f}\t"
                log_line += "\n"
                print(log_line)
                with open(log_filename, 'a') as f:
                    f.write(log_line)
                    f.write("\n")
                
                if args.evaluate_mode and args.eval_checkpoint is not None:
                    break

        # save model checkpoint after every task
        filename = "./{}seed{}_{}_{}{}_{}_acc_{:.3f}_fgt_{:.3f}_{}_lr{}_{}_sp{:.3f}_task_{}.pt".format(args.save_model,
                                                                                            seed, args.remark,
                                                                                            args.arch,
                                                                                            args.depth,
                                                                                            args.dataset,
                                                                                            prec1,
                                                                                            forgetting,
                                                                                            args.optmzr, args.lr,
                                                                                            args.lr_scheduler,
                                                                                            total_sparsity,
                                                                                            t)
        torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    main()
