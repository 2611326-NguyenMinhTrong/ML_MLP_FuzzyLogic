"""Tab 2: so sánh hai mô hình trên cùng bộ ảnh.

Trạng thái mỗi ảnh (bên phải đúng hơn hay sai hơn bên trái) được tính trực tiếp
từ cặp mô hình đang chọn, nên đổi dropdown thì màu viền đổi theo. File
bc_indices.json chỉ dùng để hiển thị số liệu trên toàn tập test làm bối cảnh.
"""

import base64
import json
from io import BytesIO

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core.inference import load_ckpt, predict, preprocess, topk
from core.registry import RESULTS_DIR, default_pair, list_checkpoints, list_samples
from ui.single_view import COARSE_HUE, CRITICAL, FINE_HUE, GOOD, MUTED, _bar, _mode

# Trạng thái so sánh giữa mô hình trái và phải trên một ảnh
FIXED = "fixed"      # trái sai -> phải đúng
BROKEN = "broken"    # trái đúng -> phải sai
SAME = "same"

STATUS_STYLE = {
    FIXED: (GOOD, "✓", "phải đúng, trái sai"),
    BROKEN: (CRITICAL, "✕", "phải sai, trái đúng"),
    SAME: (MUTED, "–", "hai bên giống nhau"),
}


@st.cache_resource(show_spinner=False)
def _model(path_str):
    return load_ckpt(path_str)


@st.cache_data(show_spinner=False)
def _predict_one(ckpt_path, img_path):
    """Dự đoán 1 ảnh bằng 1 checkpoint, có cache. Trả None nếu ảnh hỏng."""
    model, meta = _model(ckpt_path)
    try:
        r = predict(model, preprocess(Image.open(img_path), meta), meta)
    except Exception:
        return None
    return {
        "pred_fine": r["pred_fine"],
        "pred_coarse_raw": r["pred_coarse_raw"],
        "pred_hard": r["pred_hard"],
        "consistent": r["consistent"],
        "soft_viol": r["soft_viol"],
        "p_fine": r["p_fine"].tolist(),
        "p_coarse_raw": r["p_coarse_raw"].tolist(),
    }


@st.cache_data(show_spinner=False)
def _img_b64(img_path, size=96):
    """Ảnh 32×32 phóng to bằng NEAREST, mã hoá base64 để nhúng thẳng vào HTML."""
    im = Image.open(img_path).convert("RGB").resize((size, size), Image.NEAREST)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _status(left, right, true_fine):
    """Xác định trạng thái so sánh của một ảnh."""
    l_ok = left["pred_fine"] == true_fine
    r_ok = right["pred_fine"] == true_fine
    if not l_ok and r_ok:
        return FIXED
    if l_ok and not r_ok:
        return BROKEN
    return SAME


def _bc_context():
    """Đọc số liệu trên toàn tập test để hiển thị làm bối cảnh."""
    p = RESULTS_DIR / "bc_indices.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _panel(side, item, pred, meta, hue_f, hue_c):
    """Một nửa của bảng chi tiết: badge + 2 bar chart cho một mô hình."""
    import torch

    st.markdown(f"**{side} — {item['label']}**")
    ok = pred["consistent"]
    color, icon = (GOOD, "✓") if ok else (CRITICAL, "✕")
    st.markdown(
        f"<div style='background:{color};color:#fff;padding:6px 12px;"
        f"border-radius:6px;font-weight:600;font-size:14px'>"
        f"{icon} {'NHẤT QUÁN' if ok else 'VI PHẠM'}</div>",
        unsafe_allow_html=True)
    st.plotly_chart(
        _bar(topk(torch.tensor(pred["p_fine"]), meta["fine_classes"], 3),
             hue_f, "Top-3 nhãn con"),
        width="stretch", config={"displayModeBar": False},
        key=f"cmp_f_{side}")
    st.plotly_chart(
        _bar(topk(torch.tensor(pred["p_coarse_raw"]), meta["coarse_classes"], 3),
             hue_c, "Top-3 nhãn cha"),
        width="stretch", config={"displayModeBar": False},
        key=f"cmp_c_{side}")


@st.dialog("Chi tiết so sánh hai mô hình", width="large")
def _detail_dialog(row, meta, left_it, right_it, mode):
    """Hiện cửa sổ chi tiết so sánh hai mô hình trên một ảnh."""
    s = row["s"]
    fn, cn = meta["fine_classes"], meta["coarse_classes"]
    lp, rp = row["left"], row["right"]

    color, _icon, word = STATUS_STYLE[row["status"]]
    st.markdown(
        f"<img src='data:image/png;base64,{_img_b64(str(s['path']), 128)}' "
        f"style='float:left;width:96px;image-rendering:pixelated;"
        f"border:3px solid {color};border-radius:8px;margin:0 14px 6px 0'/>"
        f"<b>Ảnh:</b> <code>{s['file']}</code><br>"
        f"<b>Nhãn thật:</b> {s.get('fine_name','?')}<br>"
        f"<b>Trạng thái:</b> <span style='color:{color}'>{word}</span>",
        unsafe_allow_html=True)

    def _say(it, p):
        v = "" if p["consistent"] else ", **vi phạm**"
        return (f"{it['label']} đoán `{fn[p['pred_fine']]}` / "
                f"`{cn[p['pred_coarse_raw']]}`{v}")

    verdict = {
        FIXED: "→ **bên phải đoán đúng, bên trái đoán sai** ở ảnh này.",
        BROKEN: "→ **bên phải đoán sai, bên trái đoán đúng** ở ảnh này.",
        SAME: "→ hai mô hình **cùng đúng hoặc cùng sai** trên ảnh này.",
    }[row["status"]]
    st.markdown(f"{_say(left_it, lp)}; {_say(right_it, rp)}. {verdict}")

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        _panel("TRÁI", left_it, lp, meta, FINE_HUE[mode], COARSE_HUE[mode])
    with d2:
        _panel("PHẢI", right_it, rp, meta, FINE_HUE[mode], COARSE_HUE[mode])

    st.caption(
        f"Mức vi phạm mềm — TRÁI: **{lp['soft_viol']:.4f}** · "
        f"PHẢI: **{rp['soft_viol']:.4f}**. Fuzzy logic loss ép chỉ số này "
        "xuống, nhưng như hai biểu đồ cho thấy, giảm vi phạm không đồng nghĩa "
        "với đoán đúng hơn."
    )


def render():
    items = [d for d in list_checkpoints() if not d.get("error")]
    samples = list_samples()
    if len(items) < 2:
        st.warning("Cần ít nhất **2 checkpoint** để so sánh. "
                   "Xem README mục *Chuẩn bị dữ liệu*.")
        return
    if not samples:
        st.warning("`assets/samples/` chưa có ảnh nào.")
        return

    st.markdown(
        "Tab này đặt **hai mô hình cạnh nhau trên cùng bộ ảnh** để xem ràng "
        "buộc logic (fuzzy loss) thực sự thay đổi dự đoán ở đâu. Mặc định so "
        "**B0** (mô hình gốc, không có logic) với **M** (có logic). Mỗi ảnh "
        "được cả hai mô hình dự đoán, rồi tô viền theo việc mô hình PHẢI đúng "
        "hơn hay sai hơn mô hình TRÁI.")

    labels = [d["label"] for d in items]
    b0, m = default_pair(items)
    di = items.index(b0) if b0 in items else 0
    dj = items.index(m) if m in items else len(items) - 1

    c1, c2 = st.columns(2)
    with c1:
        i = st.selectbox("Mô hình TRÁI (thường là B0)", range(len(items)),
                         index=di, format_func=lambda k: labels[k])
    with c2:
        j = st.selectbox("Mô hình PHẢI (thường là M)", range(len(items)),
                         index=dj, format_func=lambda k: labels[k])

    if i == j:
        st.info("Đang chọn cùng một mô hình ở hai bên — mọi ảnh sẽ là *không đổi*.")

    left_it, right_it = items[i], items[j]
    _, meta = _model(str(left_it["path"]))
    mode = _mode()

    # Tính trạng thái cho toàn bộ gallery
    rows, broken = [], []
    for s in samples:
        lp = _predict_one(str(left_it["path"]), str(s["path"]))
        rp = _predict_one(str(right_it["path"]), str(s["path"]))
        if lp is None or rp is None:
            broken.append(s["file"])          # bỏ qua ảnh hỏng
            continue
        rows.append({"s": s, "left": lp, "right": rp,
                     "status": _status(lp, rp, s.get("fine"))})

    if broken:
        st.warning(f"{len(broken)} ảnh không đọc được, đã bỏ qua: "
                  f"{', '.join(broken)}")
    if not rows:
        st.error("Không có ảnh nào xử lý được trong assets/samples/.")
        return

    n_fix = sum(r["status"] == FIXED for r in rows)
    n_brk = sum(r["status"] == BROKEN for r in rows)

    # Chú giải màu
    st.markdown(
        f"<div style='display:flex;gap:20px;flex-wrap:wrap;font-size:14px;"
        f"margin:6px 0 14px'>"
        f"<span><b style='color:{GOOD}'>✓ viền xanh</b> — bên phải đúng, "
        f"bên trái sai ({n_fix} ảnh)</span>"
        f"<span><b style='color:{CRITICAL}'>✕ viền đỏ</b> — bên phải sai, "
        f"bên trái đúng ({n_brk} ảnh)</span>"
        f"<span><b style='color:{MUTED}'>– viền xám</b> — hai bên giống nhau "
        f"({len(rows) - n_fix - n_brk} ảnh)</span></div>",
        unsafe_allow_html=True)

    bc = _bc_context()
    if bc:
        st.caption(
            f"Gallery này chỉ là **{len(rows)} ảnh lấy ngẫu nhiên**. Trên toàn bộ "
            f"10.000 ảnh test (λ\\*={bc['lambda_star']:g}, seed {bc['seed']}): "
            f"**{len(bc['cell_c_fixed'])}** ảnh bên phải đúng-bên trái sai, "
            f"**{len(bc['cell_b_broken'])}** ảnh bên phải sai-bên trái đúng — tức "
            f"mô hình M (bên phải) làm sai nhiều hơn làm đúng. Xem "
            f"`table_flip_mcnemar.csv` để có kiểm định McNemar."
        )

    # Lưới ảnh
    st.markdown("**Bấm _Chi tiết_ dưới một ảnh để mở cửa sổ so sánh xác suất "
                "của hai mô hình.**")
    per_row = 5
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, row in zip(cols, rows[start:start + per_row]):
            s, status = row["s"], row["status"]
            color, icon, word = STATUS_STYLE[status]
            with col:
                st.markdown(
                    f"<div style='border:3px solid {color};border-radius:8px;"
                    f"padding:3px;line-height:0'>"
                    f"<img src='data:image/png;base64,{_img_b64(str(s['path']))}' "
                    f"style='width:100%;image-rendering:pixelated;"
                    f"border-radius:5px'/></div>"
                    f"<div style='font-size:11px;color:{MUTED};margin-top:3px;"
                    f"line-height:1.3'>{icon} {word}<br>{s.get('fine_name','?')}</div>",
                    unsafe_allow_html=True)
                if st.button("Chi tiết", key=f"pick_{s['file']}",
                             width="stretch"):
                    _detail_dialog(row, meta, left_it, right_it, mode)
