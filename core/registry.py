"""Quét thư mục checkpoints/ và sinh nhãn hiển thị cho sidebar.

Nhãn được suy ra từ `train_info` **bên trong checkpoint**, không từ tên file:
tên file có thể bị đổi, còn metadata thì đi theo trọng số.
"""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "checkpoints"
SAMPLES_DIR = PROJECT_ROOT / "assets" / "samples"
RESULTS_DIR = PROJECT_ROOT / "results"          # results.csv, bc_indices.json

# Hậu tố tên file -> mô tả biến thể (quy ước đặt tên ở mục 1.2 kế hoạch)
_SUFFIX_LABELS = {
    "_pilot": "pilot 15ep",
    "_res-product": "residuum Product",
    "_res-godel": "residuum Gödel",
    "_sq": "phạt bình phương",
    "_nowarm": "không warm-up",
}

_META_CACHE = {}


def _read_meta(path):
    """Đọc metadata của checkpoint, có cache theo (đường dẫn, thời điểm sửa)."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _META_CACHE:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        _META_CACHE[key] = {
            "train_info": ck.get("train_info", {}),
            "val_metrics": ck.get("val_metrics", {}),
            "n_fine": ck.get("config", {}).get("n_fine"),
            "n_coarse": ck.get("config", {}).get("n_coarse"),
            "arch_version": ck.get("arch_version"),
        }
    return _META_CACHE[key]


def _variant_note(stem):
    """Phần mô tả biến thể lấy từ hậu tố tên file."""
    notes = [lbl for sfx, lbl in _SUFFIX_LABELS.items() if stem.endswith(sfx)]
    return notes[0] if notes else ""


def make_label(meta, stem):
    """'B0 (λ=0, seed 0)' / 'M (λ=2, seed 0)' / 'M (λ=1, seed 0, pilot 15ep)'."""
    ti = meta["train_info"]
    lam = ti.get("lambda", 0)
    seed = ti.get("seed", "?")
    kind = "B0" if lam == 0 else "M"
    parts = [f"λ={lam:g}", f"seed {seed}"]
    note = _variant_note(stem)
    if note:
        parts.append(note)
    return f"{kind} ({', '.join(parts)})"


def list_checkpoints(ckpt_dir=None):
    """Trả list dict đã sắp xếp, mỗi dict mô tả một checkpoint dùng được.

    Checkpoint hỏng KHÔNG làm sập app — nó bị bỏ qua và kèm trường `error` để
    tầng UI hiển thị cảnh báo.
    """
    ckpt_dir = Path(ckpt_dir or CKPT_DIR)
    items = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        try:
            meta = _read_meta(p)
            ti = meta["train_info"]
            items.append({
                "path": p,
                "name": p.name,
                "label": make_label(meta, p.stem),
                "lambda": ti.get("lambda", 0),
                "seed": ti.get("seed"),
                "epochs": ti.get("epochs"),
                "residuum": ti.get("residuum"),
                "penalty": ti.get("penalty"),
                "warmup_epochs": ti.get("warmup_epochs"),
                "val_metrics": meta["val_metrics"],
                "n_fine": meta["n_fine"],
                "n_coarse": meta["n_coarse"],
                "is_baseline": ti.get("lambda", 0) == 0,
                "error": None,
            })
        except Exception as e:
            items.append({
                "path": p, "name": p.name,
                "label": f"⚠️ {p.name} (không đọc được)",
                "error": f"{type(e).__name__}: {e}",
            })

    # B0 lên đầu, rồi λ tăng dần, rồi seed
    items.sort(key=lambda d: (d.get("error") is not None,
                              d.get("lambda", 0), d.get("seed") or 0,
                              d["name"]))
    return items


def default_pair(items):
    """Cặp (B0, M) mặc định cho tab so sánh: B0 λ=0 và M có λ lớn nhất."""
    ok = [d for d in items if not d.get("error")]
    b0 = next((d for d in ok if d["is_baseline"]), None)
    m = max((d for d in ok if not d["is_baseline"]),
            key=lambda d: d["lambda"], default=None)
    return b0, m


def list_samples(samples_dir=None):
    """Danh sách ảnh mẫu trong assets/samples/ kèm chỉ số trong tập test."""
    import json
    samples_dir = Path(samples_dir or SAMPLES_DIR)
    idx_file = samples_dir / "index.json"
    if idx_file.exists():
        manifest = json.loads(idx_file.read_text(encoding="utf-8"))
        for m in manifest:
            m["path"] = samples_dir / m["file"]
        return [m for m in manifest if m["path"].exists()]
    # Không có index.json thì vẫn liệt kê được ảnh, chỉ thiếu nhãn thật
    return [{"file": p.name, "path": p, "test_index": None, "fine_name": "?"}
            for p in sorted(samples_dir.glob("*.png"))]
