import torch
import torch.nn as nn


def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class Conv2(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(Conv2, self).__init__()
        self.num_classes = num_classes

        self.convs = nn.Sequential(
            conv3x3(channels, 64, first_layer=True),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((16,16)),  # Originally nn.MaxPool2d((2,2)). If input is 32x32, this becomes equivalent.
        )

        self.output = conv1x1(256, self.num_classes)
        self.pre_linear = nn.Sequential(
            conv1x1(64 * 16 * 16, 256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.ReLU()
        )

    def forward(self, x, last=False, freeze=False):
        
        if freeze:
            with torch.no_grad():
                out = self.convs(x)
                out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
                last_lin_layer = self.pre_linear(out)
        else:
            out = self.convs(x)
            out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
            last_lin_layer = self.pre_linear(out)

        out = self.output(last_lin_layer).view(-1, self.num_classes)

        if last:
            last_lin_layer = last_lin_layer.view(-1, last_lin_layer.shape[1])
            return out, last_lin_layer
        else:
            return out

    def get_embedding_dim(self):
        return 256

    def reset(self):
        self.apply(weight_reset)


class Conv4(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(Conv4, self).__init__()
        self.num_classes = num_classes

        self.convs = nn.Sequential(
            conv3x3(channels, 64, first_layer=True),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8,8)),    # Originally nn.MaxPool2d((2,2)). If input is 32x32, this becomes equivalent.
        )

        self.output = conv1x1(256, self.num_classes)
        self.pre_linear = nn.Sequential(
            conv1x1(128 * 8 * 8, 256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.ReLU()
        )

    def forward(self, x, last=False, freeze=False):
        
        if freeze:
            with torch.no_grad():
                out = self.convs(x)
                out = out.view(out.size(0), 128 * 8 * 8, 1, 1)
                last_lin_layer = self.pre_linear(out)
        else:
            out = self.convs(x)
            out = out.view(out.size(0), 128 * 8 * 8, 1, 1)
            last_lin_layer = self.pre_linear(out)

        out = self.output(last_lin_layer).view(-1, self.num_classes)

        if last:
            last_lin_layer = last_lin_layer.view(-1, last_lin_layer.shape[1])
            return out, last_lin_layer
        else:
            return out

    def get_embedding_dim(self):
        return 256

    def reset(self):
        self.apply(weight_reset)



class Conv6(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(Conv6, self).__init__()
        self.num_classes = num_classes

        self.convs = nn.Sequential(
            conv3x3(channels, 64, first_layer=True),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(128, 256),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4,4)),    # Originally nn.MaxPool2d((2,2)). If input is 32x32, this becomes equivalent.
        )

        self.output = conv1x1(256, self.num_classes)
        self.pre_linear = nn.Sequential(
            conv1x1(256 * 4 * 4, 256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.ReLU()
        )

    def forward(self, x, last=False, freeze=False):
        
        if freeze:
            with torch.no_grad():
                out = self.convs(x)
                out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
                last_lin_layer = self.pre_linear(out)
        else:
            out = self.convs(x)
            out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
            last_lin_layer = self.pre_linear(out)

        out = self.output(last_lin_layer).view(-1, self.num_classes)

        if last:
            last_lin_layer = last_lin_layer.view(-1, last_lin_layer.shape[1])
            return out, last_lin_layer
        else:
            return out

    def get_embedding_dim(self):
        return 256

    def reset(self):
        self.apply(weight_reset)


class Conv8(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(Conv8, self).__init__()
        self.num_classes = num_classes 

        self.convs = nn.Sequential(
            conv3x3(channels, 64, first_layer=True),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(128, 256),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(256, 512),
            nn.ReLU(),
            conv3x3(512, 512),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((2,2)),    # Originally nn.MaxPool2d((2,2)). If input is 32x32, this becomes equivalent.
        )

        self.output = conv1x1(256, self.num_classes)
        self.pre_linear = nn.Sequential(
            conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.ReLU()
        )

    def forward(self, x, last=False, freeze=False):
        
        if freeze:
            with torch.no_grad():
                out = self.convs(x)
                out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
                last_lin_layer = self.pre_linear(out)
        else:
            out = self.convs(x)
            out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
            last_lin_layer = self.pre_linear(out)

        out = self.output(last_lin_layer).view(-1, self.num_classes)

        if last:
            last_lin_layer = last_lin_layer.view(-1, last_lin_layer.shape[1])
            return out, last_lin_layer
        else:
            return out

    def get_embedding_dim(self):
        return 256

    def reset(self):
        self.apply(weight_reset)



# Functions for models
def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1, first_layer=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Function for setting width of variable width networks
def scale(n, width_mult):
    return int(n * width_mult)