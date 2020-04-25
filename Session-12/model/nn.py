import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, channels):
        """
        :param channels: tuples of channels ((block1_in, block1_out), (block2_in, block2_out), ....)
        """
        super(ResBlock, self).__init__()

        conv_steps = []
        for blocks in channels:
            in_channels, out_channels = blocks
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                             padding=1,
                             stride=1)
            conv_steps.append(conv)

            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            conv_steps.append(batch_norm)

            relu = nn.ReLU()
            conv_steps.append(relu)

        self.conv_block = nn.Sequential(*conv_steps)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, out_feature=10):
        super(ResNet, self).__init__()

        self.out_feature = out_feature
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Pre Layer
        self.pre_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(block(channels=((128, 128), (128, 128))))

        # Layer 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(block(channels=((512, 512), (512, 512))))

        self.max_pool_ks4 = nn.MaxPool2d(kernel_size=4, stride=4)

        # self.fc = nn.Linear(in_features=512, out_features=out_feature)
        self.fc = nn.Conv2d(in_channels=512, out_channels=out_feature, kernel_size=1)

    def forward(self, x):
        x = self.pre_layer(x)

        x = self.layer1(x)
        b1 = self.block1(x)
        x = x + b1

        x = self.layer2(x)

        x = self.layer3(x)
        b2 = self.block2(x)
        x = x + b2

        x = self.max_pool_ks4(x)

        # x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, self.out_feature)

        return F.log_softmax(x, dim=-1)

    def summary(self, input_size=(3, 32, 32)):
        """
        Displays Model Parameters Summary
        :param input_size: input dimension for the model
        """
        summary(model=self, input_size=input_size)


if __name__ == '__main__':
    pass
