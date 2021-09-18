import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):   
    """
    from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5) #in channels, out channels (AKA # of kernels), kernel size
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # 4*4 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # max_pool2d(__,2) halves the height and width
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CustomMNISTDataset():
    """
    Check out: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for dataset info
    """
    
    def __init__(self, labels, imgs):
        temp_imgs = ((imgs-128)/128).astype(np.float32)
        self.imgs = torch.unsqueeze(torch.from_numpy(temp_imgs), 1)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return ( 
            self.imgs[idx],
            self.labels[idx]
        )
    
def multi_acc(y_pred, y_test):
    """
    multi class classification accuracy definition
    """
    
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    
    return acc
