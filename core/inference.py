"""Nạp checkpoint, tiền xử lý ảnh, và suy luận hai tầng.

Mọi tham số phụ thuộc bộ dữ liệu (mean/std, số lớp, tên lớp, ánh xạ con→cha)
đều đọc từ checkpoint chứ không đặt cứng trong app.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .model_def import ARCH_VERSION, MultiHeadMLP

# Các trường bắt buộc phải có trong file checkpoint.
REQUIRED_KEYS = (
    "arch_version", "state_dict", "config", "fine_classes", "coarse_classes",
    "fine_to_coarse", "norm_mean", "norm_std", "train_info", "val_metrics",
)


class CheckpointError(RuntimeError):
    """Lỗi checkpoint có thông điệp tiếng Việt đọc được cho người dùng cuối."""


def load_ckpt(path, device=None):
    """Nạp checkpoint, trả về (model đã eval, dict metadata).

    Kiểm tra file hợp lệ trước khi dựng model và báo lỗi rõ ràng bằng tiếng Việt.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(
            f"Không tìm thấy checkpoint: {path}\n"
            f"  -> Kiểm tra lại thư mục 'checkpoints/'. Xem README mục 'Chuẩn bị "
            f"dữ liệu' để biết cần tải/chép những file nào."
        )

    device = device or torch.device("cpu")
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:                      # file hỏng hoặc tải thiếu
        raise CheckpointError(
            f"Không đọc được checkpoint: {path.name}\n"
            f"  -> File có thể bị hỏng hoặc tải chưa xong. Chép lại file này.\n"
            f"  -> Chi tiết kỹ thuật: {type(e).__name__}: {e}"
        ) from e

    if not isinstance(ck, dict):
        raise CheckpointError(
            f"{path.name} không đúng định dạng (phải là dict).\n"
            f"  -> File này có thể được lưu bằng torch.save(model) thay vì "
            f"torch.save({{'state_dict': ...}})."
        )

    missing = [k for k in REQUIRED_KEYS if k not in ck]
    if missing:
        raise CheckpointError(
            f"{path.name} thiếu {len(missing)} trường bắt buộc: "
            f"{', '.join(missing)}\n"
            f"  -> Dùng checkpoint mới hoặc huấn luyện lại."
        )

    got = ck["arch_version"]
    if got != ARCH_VERSION:
        raise CheckpointError(
            f"Lệch phiên bản kiến trúc ở {path.name}: "
            f"checkpoint arch_version={got}, app đang dùng ARCH_VERSION={ARCH_VERSION}.\n"
            f"  -> Checkpoint sinh bởi một kiến trúc khác, nạp vào sẽ cho kết quả sai.\n"
            f"  -> Dùng checkpoint đúng phiên bản."
        )

    cfg = ck["config"]
    n_fine, n_coarse = cfg["n_fine"], cfg["n_coarse"]
    # Kiểm tra độ dài các mảng khớp với config, tránh IndexError khi suy luận.
    array_checks = [
        ("fine_classes", len(ck["fine_classes"]), n_fine),
        ("coarse_classes", len(ck["coarse_classes"]), n_coarse),
        ("fine_to_coarse", len(ck["fine_to_coarse"]), n_fine),
    ]
    bad = [(name, got, want) for name, got, want in array_checks if got != want]
    if bad:
        detail = "; ".join(f"{n}: có {g}, cần {w}" for n, g, w in bad)
        raise CheckpointError(
            f"{path.name} có metadata không nhất quán: {detail}.\n"
            f"  -> Số tên lớp không khớp config['n_fine']={n_fine}, "
            f"config['n_coarse']={n_coarse}.\n"
            f"  -> Chép lại file gốc hoặc huấn luyện lại."
        )
    if ck["fine_to_coarse"] and max(ck["fine_to_coarse"]) >= n_coarse:
        raise CheckpointError(
            f"{path.name} có fine_to_coarse trỏ ra ngoài phạm vi: giá trị lớn "
            f"nhất là {max(ck['fine_to_coarse'])} nhưng chỉ có {n_coarse} nhãn cha "
            f"(chỉ số hợp lệ 0..{n_coarse - 1})."
        )

    model = MultiHeadMLP(
        in_dim=cfg["in_dim"], hidden=tuple(cfg["hidden"]),
        n_fine=cfg["n_fine"], n_coarse=cfg["n_coarse"], p_drop=cfg["dropout"],
    ).to(device)
    try:
        model.load_state_dict(ck["state_dict"])
    except RuntimeError as e:
        raise CheckpointError(
            f"Trọng số trong {path.name} không khớp kiến trúc vừa dựng.\n"
            f"  -> Chi tiết kỹ thuật: {e}"
        ) from e

    model.eval()
    return model, ck


def preprocess(image, meta, device=None):
    """Chuyển ảnh PIL thành tensor (1, 3, 32, 32) đã chuẩn hóa.

    mean/std lấy từ metadata của checkpoint.
    """
    if not isinstance(image, Image.Image):
        raise CheckpointError("preprocess() cần một đối tượng PIL.Image.")

    # Ảnh tải lên có thể là RGBA, ảnh xám hoặc bảng màu -> ép về RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((32, 32), Image.BILINEAR)

    x = torch.from_numpy(
        __import__("numpy").asarray(image, dtype="float32")
    ).permute(2, 0, 1).div_(255.0)

    mean = torch.tensor(meta["norm_mean"]).view(3, 1, 1)
    std = torch.tensor(meta["norm_std"]).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0).to(device or torch.device("cpu"))


@torch.no_grad()
def predict(model, x, meta):
    """Suy luận hai tầng theo cả 3 chế độ raw / hard / marginal.

    Trả về dict gồm xác suất hai tầng (theo cả ba cách suy nhãn cha), nhãn dự
    đoán, cờ nhất quán, và mức vi phạm mềm soft_viol.
    """
    f2c = torch.tensor(meta["fine_to_coarse"], dtype=torch.long, device=x.device)

    logit_f, logit_c = model(x)
    p_f = F.softmax(logit_f, dim=1)[0]
    p_c = F.softmax(logit_c, dim=1)[0]

    # Marginalization: P(cha_k) = tổng P(con thuộc nhóm k)
    p_marg = torch.zeros_like(p_c)
    p_marg.index_add_(0, f2c, p_f)

    pred_f = int(p_f.argmax())
    pred_hard = int(f2c[pred_f])
    pred_c_raw = int(p_c.argmax())

    soft_viol = float(torch.relu(p_f - p_c[f2c]).sum())

    return {
        "p_fine": p_f.cpu(),
        "p_coarse_raw": p_c.cpu(),
        "p_coarse_marg": p_marg.cpu(),
        "pred_fine": pred_f,
        "pred_coarse_raw": pred_c_raw,
        "pred_hard": pred_hard,
        "pred_coarse_marg": int(p_marg.argmax()),
        "consistent": pred_hard == pred_c_raw,
        "soft_viol": soft_viol,
    }


def topk(probs, names, k=3):
    """Trả về k lớp có xác suất cao nhất dưới dạng [(tên lớp, xác suất)]."""
    k = min(k, len(names))
    vals, idx = torch.topk(probs, k)
    return [(names[int(i)], float(v)) for v, i in zip(vals, idx)]
