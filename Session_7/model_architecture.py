#Create and view model architecture
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

def model_summary(model_, input_):
    from torchsummary import summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    arch = model_.to(device)
    return summary(arch, input_)
class Net(nn.Module):
    # BN_flag 0: normal batchnorm; 1: Ghost batchnorm
    def batch_norm(self, channels, BN_flag):
        if BN_flag == 1:
            return GhostBatchNorm(channels, num_splits=2, weight=False)
        else:
            return nn.BatchNorm2d(channels)

    def __init__(self, BN_flag):
        super(Net, self).__init__()

        # Convolution Block-1 ###################################
        # Input:32x32  Outout:32x32 RF:3x3
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3, padding=1)  # input OUtput RF #28,26,3
        self.batchnorm1 = self.batch_norm(32, BN_flag=1)
        
        # Input:32x32  Outout:32x32 RF:5x5
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, padding=1)  # 26,24,5
        self.batchnorm2 = self.batch_norm(32, BN_flag=1)

        self.dp2 = nn.Dropout(0.10)
        # Transition Block-1 ###################################
        # Input:32x32  Outout:16x16 RF:10x10
        self.pool1 = nn.MaxPool2d(2, 2)  # 24,12,10

        # Input:16x16  Outout:16x16 RF:10x10
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels=16, kernel_size=1, padding=1) #20, 10, 1)  # 12,12,10
        self.batchnorm3 = self.batch_norm(16, BN_flag=1)
        self.dp3 = nn.Dropout(0.10)
      
        # Convolution Block-2 ###################################
        # Input:16x16  Outout:16x16 RF:12x12
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels=64, kernel_size=(3,1), padding=1 )  # 12,10,12
        self.batchnorm4 = self.batch_norm(64, BN_flag=1)
        self.dp4 = nn.Dropout(0.10)

        # Input:16x16  Outout:16x16 RF:14x14
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1,3), padding=1)  # 10,8,12
        self.batchnorm5 = self.batch_norm(64, BN_flag=1)
        self.dp5 = nn.Dropout(0.10)
        
        # Input:16x16  Outout:8x8 RF:28x28
        self.pool2 = nn.MaxPool2d(2, 2)  # 24,12,10

        # Transition Block-2 ###################################
        # Input:8x8  Outout:8x8 RF:28x28
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=1, padding=1) #20, 10, 1)  # 12,12,10
        self.batchnorm6 = self.batch_norm(32, BN_flag=1)
        self.dp6 = nn.Dropout(0.10)
        
        # Convolution Block-3 ###################################
        # Input:8x8  Outout:8x8 RF:30x30
        self.conv7 = nn.Conv2d(in_channels = 32, out_channels=128, dilation=2, kernel_size=3,  padding=2)  # 12,10,12
        self.batchnorm7 = self.batch_norm(128, BN_flag=1)
        self.dp7 = nn.Dropout(0.10)

        # Input:8x8  Outout:8x8 RF:32x32
        self.conv8 = nn.Conv2d(in_channels = 128, out_channels=128, dilation=2, kernel_size=3, padding=1)  # 10,8,12
        self.batchnorm8 = self.batch_norm(128, BN_flag=1)
        self.dp8 = nn.Dropout(0.10)
        
        # MP
        # Input:8x8  Outout:4x4 RF:64x64
        self.pool3 = nn.MaxPool2d(2, 2)  # 24,12,10
        
        # Input:4x4  Outout:4x4 RF:66x66
        self.conv9 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3, padding=1)  # 10,8,12
        self.conv10 = nn.Conv2d(in_channels = 256, out_channels=64, kernel_size=3, padding=1)  # 10,8,12
        
        self.avgp = nn.AvgPool2d(kernel_size=4)
        self.conv11 = nn.Conv2d(64, 10, 1)  # 6,4,18

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.dp2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.dp3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.dp4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.dp5(x)

        x = self.pool2(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.dp6(x)


        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.dp7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.dp8(x)

        x = self.pool3(x)

        x = self.conv9(x)
        x = F.relu(x)
        #x = self.batchnorm9(x)
        

        x = self.conv10(x)
        x = F.relu(x)
        #x = self.batchnorm10(x)
        #x = self.dp8(x)

        x = self.avgp(x)
        x = self.conv11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)