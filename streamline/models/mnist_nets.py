import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        

class MnistNet(nn.Module):
    def __init__(self, nclasses=10):
        super(MnistNet, self).__init__()
        self.embDim = 128
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, nclasses)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.conv1(x)
                out = F.relu(out)
                out = self.conv2(out)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
                out = self.dropout1(out)
                out = torch.flatten(out, 1)
                out = self.fc1(out)
                out = F.relu(out)
                e = self.dropout2(out) 
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = self.dropout1(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = F.relu(out)
            e = self.dropout2(out)
        out = self.fc2(e)
        if last:
            return out, e
        else:
            return out

    def get_output_dim(self):
        return self.fc2.weight.shape[0]

    def set_output_dim(self, new_dim):

        with torch.no_grad():
            old_weight_tensor = self.fc2.weight
            old_bias_tensor = self.fc2.bias

            previous_class_count = old_weight_tensor.shape[0]
            last_layer_feature_count = old_weight_tensor.shape[1]

            new_weight_tensor = torch.zeros(new_dim, old_weight_tensor.shape[1])
            new_weight_tensor = nn.init.xavier_normal_(new_weight_tensor)

            new_bias_tensor = torch.zeros(new_dim)
            new_bias_tensor = torch.flatten(nn.init.xavier_normal_(torch.stack([new_bias_tensor])))

            new_weight_tensor[:previous_class_count,:] = old_weight_tensor
            new_bias_tensor[:previous_class_count] = old_bias_tensor

            self.fc2 = nn.Linear(last_layer_feature_count,new_dim)
            self.fc2.weight.copy_(new_weight_tensor)
            self.fc2.bias.copy_(new_bias_tensor)

    def get_embedding_dim(self):
        return self.embDim

    def reset(self):
        self.apply(weight_reset)


class DeeperMnistNet(nn.Module):
    def __init__(self, nclasses=10):
        super(DeeperMnistNet, self).__init__()
        self.embDim = 128
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(15488, 128)
        self.fc2 = nn.Linear(128, nclasses)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.conv1(x)
                out = F.relu(out)
                out = self.bn1(out)
                out = self.conv2(out)
                out = F.relu(out)
                out = self.bn2(out)
                out = self.conv3(out)
                out = F.relu(out)
                out = self.bn3(out)
                out = F.max_pool2d(out, 2)
                out = self.dropout1(out)
                out = torch.flatten(out, 1)
                out = self.fc1(out)
                out = F.relu(out)
                e = self.dropout2(out) 
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = F.relu(out)
            out = self.bn3(out)
            out = F.max_pool2d(out, 2)
            out = self.dropout1(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = F.relu(out)
            e = self.dropout2(out)
        out = self.fc2(e)
        if last:
            return out, e
        else:
            return out

    def get_output_dim(self):
        return self.fc3.weight.shape[0]

    def set_output_dim(self, new_dim):

        with torch.no_grad():
            old_weight_tensor = self.fc2.weight
            old_bias_tensor = self.fc2.bias

            previous_class_count = old_weight_tensor.shape[0]
            last_layer_feature_count = old_weight_tensor.shape[1]

            new_weight_tensor = torch.zeros(new_dim, old_weight_tensor.shape[1])
            new_weight_tensor = nn.init.xavier_normal_(new_weight_tensor)

            new_bias_tensor = torch.zeros(new_dim)
            new_bias_tensor = torch.flatten(nn.init.xavier_normal_(torch.stack([new_bias_tensor])))

            new_weight_tensor[:previous_class_count,:] = old_weight_tensor
            new_bias_tensor[:previous_class_count] = old_bias_tensor

            self.fc2 = nn.Linear(last_layer_feature_count,new_dim)
            self.fc2.weight.copy_(new_weight_tensor)
            self.fc2.bias.copy_(new_bias_tensor)

    def get_embedding_dim(self):
        return self.embDim

    def reset(self):
        self.apply(weight_reset)