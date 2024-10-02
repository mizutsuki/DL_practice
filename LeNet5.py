import torch
from torch import tensor, nn,randn

class LeNet5(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(18, 54, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(1350, 84),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.seq1(x)
        x = nn.functional.softmax(x, dim=1)
        return x

test = torch.randn(1, 3, 32, 32)
le = LeNet5()(test)
cirterion = nn.CrossEntropyLoss
print(le.shape)