"""Kiến trúc mô hình MLP đa đầu ra.

Số lớp, tên lớp và mean/std chuẩn hóa được đọc từ checkpoint lúc chạy, không
đặt cứng ở đây.
"""

import torch
import torch.nn as nn

# Tăng số này nếu đổi kiến trúc, để tránh nạp nhầm checkpoint cũ.
ARCH_VERSION = 1

N_FINE, N_COARSE = 100, 20


class MultiHeadMLP(nn.Module):
    """Thân MLP dùng chung + 2 đầu ra (nhãn con / nhãn cha)."""

    def __init__(self, in_dim=3 * 32 * 32, hidden=(512, 256),
                 n_fine=N_FINE, n_coarse=N_COARSE, p_drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_fine = nn.Linear(d, n_fine)
        self.head_coarse = nn.Linear(d, n_coarse)

    def forward(self, x):
        z = self.trunk(torch.flatten(x, 1))
        return self.head_fine(z), self.head_coarse(z)
