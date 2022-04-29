import torch.nn as nn


class eye_model(nn.Module):
    """
    Model for computing single eye image:
    ConvNet tower consisting of 3 convolutional layers
    (with 7×7, 5×5 and 3×3 kernel sizes,
    strides of 2, 2 and 1,
    and 32, 64, and 128 output channels, respectively).
    ReLUs were used as non-linearities.
    """

    def __init__(self):
        super(eye_model, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm2d(32, momentum=0.9),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(64, momentum=0.9),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class landmark_model(nn.Module):
    """
    Model for the eye-corner landmarks:
    3 successive fully connected layers (with 128, 16 and 16 hidden units respectively).
    """

    def __init__(self):
        super(landmark_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x
