import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from loss.lovasz import *

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
class FocalLoss_BCE(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss_BCE, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1)

        # pt = torch.sigmoid(input)
        pt = input
        pt = pt.view(-1)
        error = torch.abs(pt - target)
        log_error = torch.log(error)
        loss = -1 * (1-error)**self.gamma * log_error
        if self.size_average: return loss.mean()
        else: return loss.sum()
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def try_bestfitting_loss(results, labels, selected_num=10):
    batch_size, class_num = results.shape
    labels = labels.view(-1, 1)
    one_hot_target = torch.zeros(batch_size, class_num + 1).cuda().scatter_(1, labels, 1)[:, :5004].contiguous()
    error_loss = lovasz_hinge(results, one_hot_target)
    labels = labels.view(-1)
    indexs_new = (labels != 5004).nonzero().view(-1)
    if len(indexs_new) == 0:
        return error_loss
    results_nonew = results[torch.arange(0, len(results))[indexs_new], labels[indexs_new]].contiguous()
    target_nonew = torch.ones_like(results_nonew).float().cuda()
    nonew_loss = nn.BCEWithLogitsLoss(reduce=True)(results_nonew, target_nonew)
    return nonew_loss + error_loss


def sigmoid_loss(results, labels, topk=10):
    if len(results.shape) == 1:
        results = results.view(1, -1)
    batch_size, class_num = results.shape
    labels = labels.view(-1, 1)
    one_hot_target = torch.zeros(batch_size, class_num + 1).cuda().scatter_(1, labels, 1)[:, :5004 * 2]
    lovasz_loss = lovasz_hinge(results, one_hot_target)
    error = torch.abs(one_hot_target - torch.sigmoid(results))
    error = error.topk(topk, 1, True, True)[0].contiguous()
    target_error = torch.zeros_like(error).float().cuda()
    error_loss = nn.BCELoss(reduce=True)(error, target_error)
    labels = labels.view(-1)
    indexs_new = (labels != 5004 * 2).nonzero().view(-1)
    if len(indexs_new) == 0:
        return error_loss
    results_nonew = results[torch.arange(0, len(results))[indexs_new], labels[indexs_new]].contiguous()
    target_nonew = torch.ones_like(results_nonew).float().cuda()
    nonew_loss = nn.BCEWithLogitsLoss(reduce=True)(results_nonew, target_nonew)
    return nonew_loss + error_loss + lovasz_loss * 0.5


def class_balanced_cross_entropy_loss(results, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(results, 0).float()
    loss_val = torch.mul(results, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(results - 2 * torch.mul(results, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(results, label):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    results = torch.sigmoid(results)

    labels_pos = torch.gt(results,0).float()
    labels_neg = torch.lt(results,1).float()

    num_labels_pos = torch.sum(labels_pos)
    num_labels_neg = torch.sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    loss_pos = torch.sum(torch.mul(labels_pos, torch.log(results + 0.00001)))
    loss_neg = torch.sum(torch.mul(labels_neg, torch.log(1 - results + 0.00001)))

    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg

    return final_loss

if __name__ == '__main__':
    results = torch.randn((4, 5004)).cuda()
    targets = torch.from_numpy(np.array([1,2,3,5004])).cuda()
    print(try_bestfitting_loss(results, targets))
