from torch import nn

padding_mode = 'replicate'


class SRCNN(nn.Module):
    def __init__(self, padding=False, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=9, padding=4*int(padding), padding_mode=padding_mode),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0),  # n1 * 1 * 1 * n2
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2*int(padding), padding_mode=padding_mode)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x