import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnWithCount(nn.Module):
    """
    CNN over packet sequence (N,2) + scalar count_norm in [0,1].
    Input:
      x_seq: (B, N, 2)
      x_count: (B, 1)
    Output:
      p_vpn: (B, 1)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64 + 1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_seq, x_count):
        # (B,N,2) -> (B,2,N)
        x = x_seq.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1)          # (B,64)

        x = torch.cat([x, x_count], dim=1)   # (B,65)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
