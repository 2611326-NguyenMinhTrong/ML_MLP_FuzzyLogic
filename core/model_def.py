"""Định nghĩa kiến trúc — BẢN SAO NGUYÊN VĂN từ PHẦN 1–2 của notebook.

⚠️ QUY TẮC VÀNG SỐ 4 (mục 1.1 kế hoạch): app KHÔNG BAO GIỜ tự định nghĩa
kiến trúc. File này là bản chép lại y hệt `MultiHeadMLP` trong
`train_model_colab/MLP_FuzzyLogic_CIFAR10.ipynb` (PHẦN 2).

Sửa kiến trúc ở notebook thì PHẢI:
  1. chép lại class xuống đây cho khớp từng dòng, và
  2. tăng ARCH_VERSION ở CẢ HAI nơi.

Nếu quên, `load_ckpt` sẽ báo lỗi lệch phiên bản thay vì nạp sai âm thầm.

Mọi thứ phụ thuộc bộ dữ liệu (số lớp, tên lớp, mean/std chuẩn hóa) KHÔNG nằm
ở đây — chúng được đọc TỪ CHECKPOINT lúc chạy. Nhờ vậy đổi CIFAR-10 sang
CIFAR-100 không phải sửa một dòng nào trong app.
"""

import torch
import torch.nn as nn

# Phải khớp ARCH_VERSION trong notebook. Lệch -> load_ckpt từ chối nạp.
ARCH_VERSION = 1

# Giá trị mặc định chỉ để tham khảo; lúc chạy luôn lấy từ checkpoint["config"].
N_FINE, N_COARSE = 100, 20


class MultiHeadMLP(nn.Module):
    """Thân MLP dùng chung + 2 đầu ra (fine / coarse)."""

    def __init__(self, in_dim=3 * 32 * 32, hidden=(512, 256),
                 n_fine=N_FINE, n_coarse=N_COARSE, p_drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        self.trunk = nn.Sequential(*layers)        # phần thân dùng chung
        self.head_fine = nn.Linear(d, n_fine)      # Head 1: nhãn con
        self.head_coarse = nn.Linear(d, n_coarse)  # Head 2: nhãn cha

    def forward(self, x):
        z = self.trunk(torch.flatten(x, 1))
        return self.head_fine(z), self.head_coarse(z)
