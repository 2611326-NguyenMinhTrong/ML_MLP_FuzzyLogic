#!/usr/bin/env python3
"""Smoke test cho phần lõi của app — DoD của Step 1.3.

Chạy:  python scripts/smoke_test.py      (exit 0 = PASS, 1 = FAIL)

Kiểm tra NGỮ NGHĨA chứ không chỉ "chạy không lỗi": các lỗi nguy hiểm nhất
trong dự án này đều không làm chương trình dừng, chúng chỉ cho ra số sai.
"""

import sys
import traceback
from pathlib import Path

# Console Windows mặc định dùng cp1252, không mã hoá được ✅/❌ hay chữ có dấu
# -> UnicodeEncodeError ngay từ dòng in đầu tiên. Ép UTF-8 ở ĐÂY, sớm nhất có
# thể (trước torch/PIL, trước mọi print), để không phải bắt người dùng nhớ
# set PYTHONIOENCODING thủ công. Phát hiện bằng cách chạy smoke test trong
# virtualenv sạch (Step 3.3 DoD) — mọi lần chạy thử trước đó trong phiên làm
# việc này đều VÔ TÌNH che mất lỗi vì luôn set biến môi trường thủ công.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.inference import CheckpointError, load_ckpt, predict, preprocess, topk
from core.model_def import ARCH_VERSION
from core.registry import list_checkpoints, list_samples

TOL = 1e-5
checks = []


def check(name, fn):
    try:
        detail = fn()
        checks.append((name, True, detail or ""))
    except Exception as e:
        checks.append((name, False, f"{type(e).__name__}: {e}"))
        if "-v" in sys.argv:
            traceback.print_exc()


ctx = {}


def t1_registry():
    items = list_checkpoints()
    ok = [d for d in items if not d.get("error")]
    assert ok, ("không tìm thấy checkpoint hợp lệ nào trong checkpoints/ "
                "— xem README mục 'Chuẩn bị dữ liệu'")
    bad = [d for d in items if d.get("error")]
    ctx["items"] = ok
    return (f"{len(ok)} checkpoint hợp lệ"
            + (f", {len(bad)} lỗi (đã bỏ qua, không làm sập app)" if bad else ""))


def t2_labels():
    labels = [d["label"] for d in ctx["items"]]
    assert len(labels) == len(set(labels)), f"nhãn hiển thị bị trùng: {labels}"
    assert any(d["is_baseline"] for d in ctx["items"]), "không có mốc B0 (λ=0)"
    return " | ".join(labels[:3]) + (" ..." if len(labels) > 3 else "")


def t3_samples():
    s = list_samples()
    assert s, "assets/samples/ rỗng — cần 20 ảnh PNG do PHẦN 6 notebook xuất"
    ctx["samples"] = s
    with_idx = sum(1 for m in s if m.get("test_index") is not None)
    return f"{len(s)} ảnh, {with_idx} ảnh có chỉ số tập test (khớp bc_indices)"


def t4_load():
    model, meta = load_ckpt(ctx["items"][0]["path"])
    assert meta["arch_version"] == ARCH_VERSION
    n_f, n_c = meta["config"]["n_fine"], meta["config"]["n_coarse"]
    assert len(meta["fine_classes"]) == n_f, "số tên lớp con lệch config"
    assert len(meta["coarse_classes"]) == n_c, "số tên lớp cha lệch config"
    assert len(meta["fine_to_coarse"]) == n_f, "ánh xạ con->cha lệch số lớp con"
    assert max(meta["fine_to_coarse"]) < n_c, "ánh xạ trỏ ra ngoài số lớp cha"
    ctx["model"], ctx["meta"] = model, meta
    return f"{n_f} nhãn con -> {n_c} nhãn cha, arch_version={ARCH_VERSION}"


def t5_preprocess():
    meta = ctx["meta"]
    img = Image.open(ctx["samples"][0]["path"])
    x = preprocess(img, meta)
    assert tuple(x.shape) == (1, 3, 32, 32), f"shape sai: {tuple(x.shape)}"
    ctx["x"] = x
    # Ảnh RGBA / xám / bảng màu đều phải nuốt được, không được ném lỗi
    for mode in ("RGBA", "L", "P"):
        preprocess(img.convert(mode), meta)
    return f"{tuple(x.shape)}, nuốt được cả RGBA/L/P"


def t6_norm_from_ckpt():
    """mean/std PHẢI đọc từ checkpoint, không hard-code."""
    meta = dict(ctx["meta"])
    img = Image.open(ctx["samples"][0]["path"])
    x1 = preprocess(img, meta)
    meta["norm_mean"] = [0.0, 0.0, 0.0]
    meta["norm_std"] = [1.0, 1.0, 1.0]
    x2 = preprocess(img, meta)
    assert not torch.allclose(x1, x2), (
        "đổi norm_mean/std trong metadata mà kết quả không đổi "
        "-> preprocess đang hard-code hằng số chuẩn hóa"
    )
    return "đổi mean/std trong metadata thì đầu ra đổi theo"


def t7_predict():
    r = predict(ctx["model"], ctx["x"], ctx["meta"])
    meta = ctx["meta"]
    n_f, n_c = meta["config"]["n_fine"], meta["config"]["n_coarse"]

    assert r["p_fine"].shape == (n_f,), "p_fine sai kích thước"
    assert r["p_coarse_raw"].shape == (n_c,), "p_coarse_raw sai kích thước"
    for k in ("p_fine", "p_coarse_raw", "p_coarse_marg"):
        s = float(r[k].sum())
        assert abs(s - 1.0) < TOL, f"{k} không tổng bằng 1 (được {s:.6f})"

    f2c = meta["fine_to_coarse"]
    assert r["pred_hard"] == f2c[r["pred_fine"]], \
        "pred_hard phải bằng cha(argmax nhãn con)"
    assert r["consistent"] == (r["pred_hard"] == r["pred_coarse_raw"]), \
        "cờ consistent không khớp định nghĩa"
    assert isinstance(r["consistent"], bool)
    assert r["soft_viol"] >= 0, "soft_viol không thể âm"
    ctx["r"] = r
    return (f"{meta['fine_classes'][r['pred_fine']]} / "
            f"{meta['coarse_classes'][r['pred_coarse_raw']]}, "
            f"{'NHẤT QUÁN' if r['consistent'] else 'VI PHẠM'}")


def t8_marginal_math():
    """P(cha_k) của chế độ marginal phải đúng bằng tổng P(con) trong nhóm k."""
    r, meta = ctx["r"], ctx["meta"]
    f2c = torch.tensor(meta["fine_to_coarse"])
    manual = torch.zeros(meta["config"]["n_coarse"])
    manual.index_add_(0, f2c, r["p_fine"])
    assert torch.allclose(manual, r["p_coarse_marg"], atol=TOL), \
        "p_coarse_marg không bằng tổng xác suất các nhãn con cùng nhóm"
    return "khớp tổng thủ công"


def t9_hard_always_consistent():
    """Chế độ hard nhất quán 100% theo cách xây dựng — đúng với MỌI ảnh mẫu."""
    n = 0
    for s in ctx["samples"][:10]:
        x = preprocess(Image.open(s["path"]), ctx["meta"])
        r = predict(ctx["model"], x, ctx["meta"])
        assert r["pred_hard"] == ctx["meta"]["fine_to_coarse"][r["pred_fine"]]
        n += 1
    return f"đúng trên {n}/{n} ảnh"


def t10_arch_guard():
    """Lệch arch_version phải bị TỪ CHỐI, không được nạp im lặng."""
    import tempfile
    src = ctx["items"][0]["path"]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["arch_version"] = 999
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.pt"
        torch.save(ck, bad)
        try:
            load_ckpt(bad)
        except CheckpointError as e:
            assert "arch_version" in str(e)
            return "báo lỗi thân thiện đúng như mong đợi"
        raise AssertionError("load_ckpt đã nạp checkpoint lệch phiên bản!")


def t11_missing_file():
    """File không tồn tại -> CheckpointError có hướng dẫn, không phải traceback."""
    try:
        load_ckpt(Path("checkpoints") / "khong_ton_tai.pt")
    except CheckpointError as e:
        assert "README" in str(e) or "checkpoints" in str(e)
        return "có thông điệp hướng dẫn người dùng"
    raise AssertionError("phải báo CheckpointError khi thiếu file")


def t12_topk():
    top = topk(ctx["r"]["p_fine"], ctx["meta"]["fine_classes"], k=3)
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1], "top-k chưa sắp giảm dần"
    return ", ".join(f"{n} {p:.1%}" for n, p in top)


for name, fn in [
    ("1. registry quét được checkpoint", t1_registry),
    ("2. nhãn hiển thị không trùng, có B0", t2_labels),
    ("3. có ảnh mẫu trong assets/samples", t3_samples),
    ("4. load_ckpt + contract nhất quán", t4_load),
    ("5. preprocess ra tensor 3x32x32", t5_preprocess),
    ("6. mean/std đọc TỪ checkpoint", t6_norm_from_ckpt),
    ("7. predict trả đủ 3 chế độ", t7_predict),
    ("8. marginal = tổng xác suất nhóm", t8_marginal_math),
    ("9. hard luôn nhất quán 100%", t9_hard_always_consistent),
    ("10. chặn checkpoint lệch arch_version", t10_arch_guard),
    ("11. báo lỗi thân thiện khi thiếu file", t11_missing_file),
    ("12. top-k sắp xếp đúng", t12_topk),
]:
    check(name, fn)

print("=" * 68)
print(f"{'SMOKE TEST — fuzzy-mlp-demo (Step 1.3)':^68}")
print("=" * 68)
for name, ok, detail in checks:
    print(f"  [{'PASS ✅' if ok else 'FAIL ❌'}] {name}")
    if detail:
        print(f"           {detail}")

n_ok = sum(1 for _, ok, _ in checks if ok)
print("-" * 68)
if n_ok == len(checks):
    print(f"  ✅ PASS — {n_ok}/{len(checks)} kiểm tra xanh")
    sys.exit(0)
print(f"  ❌ FAIL — {n_ok}/{len(checks)} xanh. Chạy lại với -v để xem traceback.")
sys.exit(1)
