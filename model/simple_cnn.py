import torch.nn as nn


class Simple_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channel = config.IN_CHANNEL      
        out_channel = config.OUT_CHANNEL    
        kernel_size = config.KERNEL_SIZE    
        num_classes = config.NUM_CLASSES 

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * 2, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel * 4, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel * 4, out_channel * 4, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(out_channel * 4 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.block_1(x)  # [B, 32, 112, 112]
        x = self.block_2(x)  # [B, 64, 56, 56]
        x = self.block_3(x)  # [B, 128, 28, 28]
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)
