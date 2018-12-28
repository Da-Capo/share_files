labels = ['LCZ1-Compact high-rise ',
         'LCZ2-Compact mid-rise ',
         'LCZ3-Compact low-rise ',
         'LCZ4-Open high-rise ',
         'LCZ5-Open mid-rise ',
         'LCZ6-Open low-rise ',
         'LCZ7-Lightweight low-rise ',
         'LCZ8-Large low-rise',
         'LCZ9-Sparse low-rise',
         'LCZ10-Heavy industry',
         'LCZ A-dense trees',
         'LCZ B-Scattered trees',
         'LCZ C-Bush/scrub',
         'LCZ D-Low plants',
         'LCZ E-Bare rock/paved',
         'LCZ F-Bare soil/sand',
         'LCZ G-Water']





#@title models
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 18
        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
#         out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes):

    return ResNet(ResidualBlock, num_classes)
  
  

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(18, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def Densenet121(num_classes, **kwargs):
    model = DenseNet(num_classes=num_classes, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model
  
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['se_resnext26_32x4d', 'se_resnext50_32x4d', 'se_resnext101_32x4d']


class SEBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(SEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes * 4, planes // 4)
        self.fc2 = nn.Linear(planes // 4, planes * 4)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        se = self.global_avg(out)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)

        out = out * se.expand_as(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SE_ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, baseWidth=4, cardinality=32, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000):
        """ Constructor
        Args:
            baseWidth: baseWidth for SE_ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(SE_ResNeXt, self).__init__()
        block = SEBottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(18, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(18, 32, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, groups=8, bias=False)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, groups=16, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct SE_ResNeXt
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))
            
        layers.append(nn.Dropout(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnext(baseWidth=4, cardinality=32, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000):
    """
    Construct SE_ResNeXt.
    (2, 2, 2, 2) for se_resnext26
    (3, 4, 6, 3) for se_resnext50
    (3, 4, 23, 3) for se_resnext101
    (3, 8, 36, 3) for se_resnext152
    """
    model = SE_ResNeXt(baseWidth=baseWidth, cardinality=cardinality, head7x7=head7x7,
                       layers=layers, num_classes=num_classes)
    return model


def se_resnext26_32x4d(num_classes):
    model = SE_ResNeXt(baseWidth=4, cardinality=32, head7x7=False, layers=(2, 2, 2, 2), num_classes=num_classes)
    return model


def se_resnext50_32x4d(num_classes):
    model = SE_ResNeXt(baseWidth=4, cardinality=32, head7x7=False, layers=(3, 4, 6, 3), num_classes=num_classes)
    return model


def se_resnext101_32x4d(num_classes):
    model = SE_ResNeXt(baseWidth=4, cardinality=32, head7x7=False, layers=(3, 4, 23, 3), num_classes=num_classes)
    return model
  
  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
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
  
class BaseNet(nn.Module):
    def __init__(self, num_classes=17):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(18, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.fc1 = nn.Linear(4*4*256, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.activate = nn.PReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.activate(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.activate(self.conv3(x))
        x = x.view(-1, 4*4*256)
        x = self.activate(self.fc1(x))
        x = self.dropout(x)
        x = self.activate(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    

###########################################################################################################################
# @title train&val&predict function
    
def validate(val_loader, model, criterion, helper):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # Validation
    model.eval()
    with torch.set_grad_enabled(False):
        for i, (local_batch, local_labels) in enumerate(val_loader):
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            output = model(local_batch)
            loss = criterion(output, local_labels)

            # measure utils.accuracy and record loss
            try:
                prec1, prec3 = accuracy(output.data, local_labels, topk=(1, 3))
            except:
                prec1, prec3 = accuracy(output.data, local_labels, topk=(1, 1))
            losses.update(float(loss), local_batch.size(0))
            top1.update(float(prec1), local_batch.size(0))
            top3.update(float(prec3), local_batch.size(0))

#         helper.log(' * Prec@1 {top1.avg:.3f} Loss {losses.avg:.3f}'
#                    .format(top1=top1, losses=losses))
        outlog = ('valLoss {losses.avg:.3f} valPrec@1 {top1.avg:.3f}'
                   .format(top1=top1, losses=losses))
        
    return losses.avg, top1.avg, outlog


def train(train_loader, model, criterion, optimizer, scheduler, epoch, helper):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # switch to train mode
    model.train()

    # Training
    for i, (local_batch, local_labels) in enumerate(train_loader):
        local_batch, local_labels = local_batch.to(
            device), local_labels.to(device)
        output = model(local_batch)
        loss = criterion(output, local_labels)
        try:
            prec1, prec3 = accuracy(output.data, local_labels, topk=(1, 3))
        except:
            prec1, prec3 = accuracy(output.data, local_labels, topk=(1, 1))
        losses.update(float(loss), local_batch.size(0))
        top1.update(float(prec1), local_batch.size(0))
        top3.update(float(prec3), local_batch.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        for group in optimizer.param_groups:
            lr_cur = group['lr']
#         if (i+1) % 20 == 0:
#             helper.log('Epoch: [{0}][{1}/{2}]\t'
#                   'Lr: {lr:.4e}\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
#                       epoch, i, len(train_loader), lr=lr_cur, loss=losses, top1=top1, top3=top3))

#     helper.log('Epoch: [{0}][{1}/{2}]\t'
#                'Lr: {lr:.4e}\t'
#                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
#                    epoch, i, len(train_loader), lr=lr_cur, loss=losses, top1=top1, top3=top3))
    outlog = ('Epoch: [{0}]\t'
               'Lr: {lr:.4e}\t'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, lr=lr_cur, loss=losses, top1=top1))
#     helper.log(outlog)
    return outlog


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


class EarlyStop(object):
    # if earlystop.step(val):break
    def __init__(self, patience, mode="max"):
        assert mode in ["min", "max"]
        self.patience = patience
        self.mode = mode
        self.i = 0
        self.best_val = None

    def is_better(self, val):
        if not self.best_val:
            self.best_val = val
            return True

        if self.mode == "min":
            return self.best_val > val
        if self.mode == "max":
            return self.best_val < val

    def is_stop(self, val):
        if self.is_better(val):
            self.best_val = val
            self.i = 0
        else:
            self.i += 1
            if self.i > self.patience:
                return True

        return False


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


def plot_confusion_matrix(y_true, y_pred, labels,  title="confusion_matrix", save_path=""):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    plt.grid(False)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0  # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red',
                     fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                # 这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red',
                         fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red',
                         fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig(save_path+"/" + title + '.jpg', dpi=300)
    plt.show()


def predict(model, loader):
    model.eval()
    outputs = []
    with torch.set_grad_enabled(False):
        for i, (local_batch, local_labels) in enumerate(loader):
            output = model(local_batch.to(device)).cpu().numpy()
            outputs.append(output)
    return np.argmax(np.concatenate(outputs, 0), -1), np.concatenate(outputs, 0)


def stacking_predict(model_paths, loader):
    sum_ouput = np.zeros((len(loader.dataset), 17))
    for model_path in model_paths:
        model = torch.load(model_path)
        torch.cuda.empty_cache()
        model.eval()
        outputs = []
        with torch.set_grad_enabled(False):
            for i, (local_batch, local_labels) in enumerate(loader):
                output = model(local_batch.to(device)).cpu().numpy()
                outputs.append(output)

        dist_pred = np.exp(np.concatenate(outputs, 0))
        e_x = np.exp(dist_pred - np.max(dist_pred, axis=1, keepdims=True))
        sm = e_x/e_x.sum(1, keepdims=True)
        sum_ouput += sm
    return np.argmax(sum_ouput, -1), sm


def stacking_outputs_list(outputs_list):
    sum_output = np.zeros(outputs_list[0].shape)
    for outputs in outputs_list:
        dist_pred = np.exp(outputs)
        e_x = np.exp(dist_pred - np.max(dist_pred, axis=1, keepdims=True))
        sm = e_x/e_x.sum(1, keepdims=True)
        sum_output += sm
    return np.argmax(sum_output, -1), sm


def smooth_loss(pred, gold):
    gold = gold.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.mean()
    return loss


###########################################################################################################################
# @title  helper functions
class Dataset(data.Dataset):
    def __init__(self, fid, class_config=None, aug=False):
        try:
            class_map = self.get_class_map(class_config)
            labels = [class_map.get(l, -1) for l in np.argmax(fid['label'], -1).tolist()]
            partition = [i for i, l in enumerate(labels) if l != -1]
        except:
            labels = np.zeros(len(fid['sen1']))
            partition = np.arange(0, len(fid['sen1'])).tolist()

        self.labels = labels
        self.list_IDs = partition
        self.fid = fid
        self.aug = aug

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = torch.tensor(self.get_tfms_img(ID), dtype=torch.float).permute(2, 0, 1)
        y = torch.tensor(self.labels[ID], dtype=torch.long)
        return X, y
    

    def get_tfms_img(self, ID):
        img = np.concatenate([self.fid['sen1'][ID], self.fid['sen2'][ID]], axis=-1).astype(np.float32)
        img -= mean
        img /= std
        img = np.clip(img,tmin,tmax)/(tmax-tmin)
        if self.aug:
            if np.random.random_sample() > 0.5:
                img = cv2.flip(img, 0)
            if np.random.random_sample() > 0.5:
                img = cv2.flip(img, 1)
        return img
    
    def get_class_map(self, class_config):
        if not class_config:
            class_config = np.arange(17).reshape(17, 1).tolist()
        class_map = {}
        for idx, cc in enumerate(class_config):
            for c in cc:
                class_map[c] = idx
        self.num_class = len(class_config)
        return class_map
    
    def get_targets(self):
        return np.array(self.labels)[np.array(self.list_IDs)].tolist()

    
    def avg_sampler(self, num_samples):
        targets = self.get_targets()
        counter = Counter(targets)
        weights = [len(targets)/counter[l] for l in targets]
        sampler = data.sampler.WeightedRandomSampler(weights,
                                                     num_samples=num_samples,
                                                     replacement=True)
        return sampler
    
    def random_sampler(self, num_samples):
        sampler = data.sampler.RandomSampler(self, num_samples=num_samples,
                                                     replacement=True)
        return sampler

    def set_kfold(self, k, n_splits, dtype="train"):
        from sklearn.model_selection import StratifiedKFold
        X, y = np.array(self.list_IDs), np.array(self.get_targets())
        skf = StratifiedKFold(n_splits=n_splits, random_state=2048, shuffle=True)

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            if idx == k:
                if dtype == "train":
                    self.list_IDs = X[train_index].tolist()
                if dtype == "val":
                    self.list_IDs = X[test_index].tolist()

class kfDataset(Dataset):
    def __init__(self, fid,  kf, n_splits, dtype="train", class_config=None, aug=False):
        try:
            class_map = self.get_class_map(class_config)
            labels = [class_map.get(l, -1) for l in np.argmax(fid['label'], -1).tolist()]
            partition = [i for i, l in enumerate(labels) if l != -1]
        except:
            labels = np.zeros(len(fid['sen1']))
            partition = np.arange(0, len(fid['sen1'])).tolist()

        self.labels = labels
        self.list_IDs = partition
        self.fid = fid
        self.aug = aug
        
        from sklearn.model_selection import StratifiedKFold
        X, y = np.array(self.list_IDs), np.array(self.get_targets())
        skf = StratifiedKFold(
            n_splits=n_splits, random_state=2048, shuffle=True)

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            if idx == kf:
                if dtype == "train":
                    self.list_IDs = X[train_index].tolist()
                if dtype == "val":
                    self.list_IDs = X[test_index].tolist()
                    
    def acc(self, y_pred):
        pred_arr = np.array(y_pred)[np.array(self.list_IDs)]
        return np.sum(pred_arr==self.get_targets())/len(self.list_IDs)

class Helper:
    def __init__(self, work_dir, model_name):
        self.work_dir = work_dir+"/"
        self.model_name = model_name
        try:
            os.stat(self.work_dir)
        except:
            print("making dir:", self.work_dir)
            os.mkdir(self.work_dir)

    def log(self, string):
        string = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime())+" "+str(string)
        print(string)
        with open(self.work_dir+"log.txt", "a") as f:
            f.write(string+"\n")

    def get_model(self, name, num_classes=17):
        helper.log("Model: %s %s" % (model_name, num_classes))
        if name == "basenet":
            model = BaseNet(num_classes=num_classes).to(device)
        if name == "resnet18":
            model = ResNet18(num_classes=num_classes).to(device)
        if name == "densenet121":
            model = Densenet121(num_classes, drop_rate=0.2).to(device)
        if name == "seresnext26":
            model = se_resnext26_32x4d(num_classes=num_classes).to(device)
        if name == "seresnext50":
            model = se_resnext50_32x4d(num_classes=num_classes).to(device)
        if name == "seresnext101":
            model = se_resnext101_32x4d(num_classes=num_classes).to(device)
        return model, -1

    def load_model(self, rule):
        import glob
        model_path = glob.glob(self.work_dir+rule)[-1]
        n_ep = int(model_path.split('.')[-2][2:])
        print("finded :", model_path, n_ep)
        return torch.load(model_path), n_ep

    def save_model(self, model, filename, epoch=None):
        if epoch:
            path = self.work_dir+'%s_%s_%s.ep%d.pt' % (time.strftime(
                "%Y%m%d%H%M", time.localtime()), filename, self.model_name, epoch)
        else:
            path = self.work_dir+'%s.pt' % (filename)
#     self.log("saved: %s"%path)
        torch.save(model, path)

    def plot_cm(self, model, loader, outfile):
        y_pred, _ = predict(model, loader)
        y_true = loader.dataset.get_targets()
        plot_confusion_matrix(y_true,
                              y_pred,
                              range(loader.dataset.num_class),
                              title=outfile,
                              save_path=self.work_dir)

    def save_output(self, outfile, ouputs):
        path = self.work_dir+str(outfile)+".npz"
        self.log("saved: %s" % path)
        np.savez(path, ouputs)
    
