import os
import torch
import torch.nn as nn
import torch.optim as optim

import nets


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

class BasicBlockReduced(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockReduced, self).__init__()
        # Adjust conv layers to have fewer channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetReduced(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetReduced, self).__init__()
        self.in_channels = 32  # Reduced initial channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # We have observed that the images in the CIFAR-10 dataset have a resolution of 32x32 pixels, 
        # which is relatively small compared to other datasets. Consequently, to retain more information during training, 
        # it is advisable to utilize smaller kernel sizes and remove maxpooling from the network architecture.
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)  # Adjusted based on final layer channels

    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    def resnet18_reduced(**kwargs):
        # Adjust the layer configuration to reduce complexity
        model = ResNetReduced(BasicBlockReduced, [2, 2, 2, 2], **kwargs)
        return model

class ResNetModel:
    def __init__(self, opt, train=True):
        # Initialize the reduced ResNet model.
        self.net = ResNetReduced(BasicBlockReduced, [2, 2, 2, 2], num_classes=10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.net.to(self.device)

        # Use DataParallel for multi-GPU setups.
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)

        init_weights(self.net)

        # Print the total number of parameters in the model.
        num_params = sum(p.numel() for p in self.net.parameters())
        print(f'Total number of parameters : {num_params / 1e6:.3f} M')

        self.checkpoint_dir = opt.checkpoint_dir if train else None
        if train:
            self.net.train()
            self.set_optimizer(opt)
            self.criterion = nn.CrossEntropyLoss()
            self.loss = 0.0
        else:
            self.net.eval()

    def set_optimizer(self, opt):
        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=opt.lr,
                momentum=opt.momentum if 'momentum' in opt else 0,
                weight_decay=opt.weight_decay if 'weight_decay' in opt else 0
            )
        elif opt.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr)
        else:
            raise ValueError("Unsupported optimizer")

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=opt.decay_milestones,
            gamma=opt.lr_decay_rate
        )
        
        # Add dynamic learning scheduler to help the model converge
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def optimize_params(self, x, label):
        x, label = x.to(self.device), label.to(self.device)
        self.optimizer.zero_grad()
        y = self._forward(x)
        self.loss = self.criterion(y, label)
        self.loss.backward()
        self.optimizer.step()

    def _forward(self, x):
        return self.net(x)

    def test(self, x, label):
        x, label = x.to(self.device), label.to(self.device)
        with torch.no_grad():
            outputs = self._forward(x)
            _, predicted = torch.max(outputs, 1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
        return correct, total, predicted

            
            
    def val(self, x, label):
        x, label = x.to(self.device), label.to(self.device)
        with torch.no_grad():
            y = self._forward(x)
            loss = self.criterion(y, label)
        return loss.item()

    def save_model(self, name):
        path = os.path.join(self.checkpoint_dir, f'model_{name}.pth')
        torch.save(self.net.state_dict(), path)
        print(f'model saved to {path}')


    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
        print(f'model loaded from {path}')


    def get_current_loss(self):
        return self.loss.item()

