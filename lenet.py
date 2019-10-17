import torch.nn as nn
import math

class LeNet(nn.Module):
    def __init__(self, input_channel):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, 6, 5, 1, 2),
                                   nn.ReLU(inplace = True),
                                   #nn.BatchNorm2d(6),
                                   nn.Conv2d(6, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d(2))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   #nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 84, 3, 1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(84,120, 5, 1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(120), )
        self.fc1 = nn.Sequential(nn.Linear(120, 84),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(84, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
