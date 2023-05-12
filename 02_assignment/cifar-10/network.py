import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # ----------TODO------------
        # define a network 
        # ----------TODO------------
        '''#一个大约能65%的模型
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2, stride=1, dilation=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )'''
        # 最终使用的模型 能达79%左右
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,3),   #20x3x32x32 -> 20x32x30x30
            nn.BatchNorm2d(32),  #加了个正则化层
            nn.ReLU(),     
            nn.Conv2d(32,32,3),   # ->20x32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), #20x32x14x14

            nn.Conv2d(32,64,3), #->20x64x12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3), #->20x64x10x10
            nn.BatchNorm2d(64), 
            nn.ReLU(),

            nn.MaxPool2d((2,2)), #20x64x5x5
            nn.Flatten(), #20x64x5x5
            nn.Linear(1600,512), #20x1600->20x512
            nn.ReLU(), #20x512
            nn.Linear(512,10), #20x512->20x10
        )

        
    def forward(self, x):
        # ----------TODO------------
        # network forwarding 
        # ----------TODO------------
        out = self.layer1(x)
        return out



if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
