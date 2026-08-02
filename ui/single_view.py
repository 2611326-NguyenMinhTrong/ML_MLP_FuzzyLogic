"""Tab 1: xem dự đoán của một mô hình trên một ảnh.

Cột trái chọn ảnh (gallery hoặc upload), cột phải hiển thị badge nhất quán,
top-3 hai tầng và bảng so ba chế độ suy luận.
"""

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core.inference import predict, preprocess, topk
from core.registry import list_samples

# Màu cho hai biểu đồ (nhãn con xanh dương, nhãn cha xanh lá) theo chế độ sáng/tối
FINE_HUE = {"light": "#2a78d6", "dark": "#3987e5"}
COARSE_HUE = {"light": "#008300", "dark": "#008300"}
MUTED = "#898781"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


def _mode():
    """Chế độ sáng/tối hiện tại của Streamlit, mặc định 'light'."""
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def _bar(items, hue, title):
    """Vẽ bar chart ngang cho top-k, ghi giá trị trực tiếp ở đầu mỗi thanh."""
    names = [n for n, _ in items][::-1]      # plotly vẽ từ dưới lên
    vals = [v * 100 for _, v in items][::-1]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=hue, cornerradius=4),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color=MUTED, size=12),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        height=150 + 26 * len(items),
        # automargin để tên lớp dài không bị cắt
        margin=dict(l=4, r=56, t=34, b=0, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
        showlegend=False,
        xaxis=dict(visible=False, range=[0, max(vals) * 1.28 or 1]),
        yaxis=dict(tickfont=dict(color=MUTED, size=12), ticksuffix="  ",
                   automargin=True),
    )
    return fig


def _pick_image(samples):
    """Trả (PIL image, mô tả nguồn, có phải ảnh ngoài phân bố không)."""
    src = st.radio("Nguồn ảnh", ["Gallery CIFAR-100", "Tải ảnh lên"],
                   horizontal=True, label_visibility="collapsed")

    if src == "Gallery CIFAR-100":
        if not samples:
            st.warning("`assets/samples/` chưa có ảnh nào. "
                       "Xem README mục *Chuẩn bị dữ liệu*.")
            return None, "", False
        labels = [f"{m.get('fine_name', '?')} — {m['file']}" for m in samples]
        i = st.selectbox("Chọn ảnh mẫu", range(len(samples)),
                         format_func=lambda k: labels[k])
        m = samples[i]
        img = _open_and_validate(m["path"], f"`{m['file']}`")
        if img is None:
            return None, "", False
        return img, f"nhãn thật: **{m.get('fine_name', '?')}**", False

    up = st.file_uploader("Chọn tệp ảnh", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if up is None:
        st.info("Chọn một ảnh để bắt đầu, hoặc chuyển sang tab Gallery.")
        return None, "", False
    img = _open_and_validate(up, "tệp bạn vừa tải lên")
    return (img, "ảnh tải lên", True) if img else (None, "", False)


def _open_and_validate(source, label):
    """Mở ảnh và ép giải mã ngay để phát hiện file hỏng sớm.

    PIL nạp ảnh kiểu lười, file cắt cụt chỉ báo lỗi khi đụng tới pixel, nên gọi
    img.load() ngay tại đây và bắt lỗi luôn.
    """
    try:
        img = Image.open(source)
        img.load()
        return img
    except Exception as e:
        st.error(f"Không đọc được {label}: {type(e).__name__}: {e}\n\n"
                 "Tệp có thể bị hỏng hoặc tải lên chưa xong. Thử ảnh khác.")
        return None


def render(model, meta):
    samples = list_samples()
    mode = _mode()
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        img, caption, is_external = _pick_image(samples)
        if img is None:
            return

        a, b = st.columns(2)
        with a:
            st.image(img, caption="Ảnh gốc", width='stretch')
        with b:
            # Phóng to bản 32x32 bằng NEAREST để thấy rõ từng pixel
            small = img.convert("RGB").resize((32, 32), Image.BILINEAR)
            st.image(small.resize((256, 256), Image.NEAREST),
                     caption="Thứ mô hình nhìn thấy (32×32)",
                     width='stretch')
        if caption:
            st.caption(caption)

        if is_external:
            st.warning(
                "**Ảnh ngoài phân bố CIFAR-100.** Mô hình là MLP thuần huấn "
                "luyện trên ảnh 32×32 và chỉ đạt ~25% độ chính xác nhãn con, "
                "nên ảnh từ Internet sẽ sai rất thường xuyên. Dùng gallery để "
                "xem cơ chế hai tầng đúng như thiết kế."
            )

    # Bọc try/except quanh phần xử lý ảnh để phòng ảnh hỏng.
    try:
        x = preprocess(img, meta)
        r = predict(model, x, meta)
    except Exception as e:
        st.error(
            f"Không xử lý được ảnh này: {type(e).__name__}: {e}\n\n"
            "Tệp có thể bị hỏng hoặc tải lên chưa xong. Thử lại với một ảnh khác."
        )
        return

    fine_names = meta["fine_classes"]
    coarse_names = meta["coarse_classes"]

    with right:
        # Badge nhất quán / vi phạm
        if r["consistent"]:
            st.markdown(
                f"<div style='background:{GOOD};color:#fff;padding:10px 16px;"
                f"border-radius:8px;font-size:17px;font-weight:600'>"
                f"✓ NHẤT QUÁN — nhãn con và nhãn cha khớp nhau</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:{CRITICAL};color:#fff;padding:10px 16px;"
                f"border-radius:8px;font-size:17px;font-weight:600'>"
                f"✕ VI PHẠM — hai tầng mâu thuẫn nhau</div>",
                unsafe_allow_html=True)
        st.caption(
            f"`{fine_names[r['pred_fine']]}` thuộc nhóm "
            f"`{coarse_names[r['pred_hard']]}`, nhưng đầu ra tầng cha nói "
            f"`{coarse_names[r['pred_coarse_raw']]}`."
            if not r["consistent"] else
            f"`{fine_names[r['pred_fine']]}` → `{coarse_names[r['pred_hard']]}`, "
            f"và đầu ra tầng cha cũng nói vậy."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _bar(topk(r["p_fine"], fine_names, 3), FINE_HUE[mode],
                     "Top-3 nhãn con"),
                width='stretch', config={"displayModeBar": False})
        with c2:
            st.plotly_chart(
                _bar(topk(r["p_coarse_raw"], coarse_names, 3), COARSE_HUE[mode],
                     "Top-3 nhãn cha (đầu ra thô)"),
                width='stretch', config={"displayModeBar": False})

        # --- Bảng so ba chế độ suy luận ---
        st.markdown("**Ba chế độ suy ra nhãn cha**")
        rows = [
            {"Chế độ": "raw", "Cách làm": "argmax đầu ra tầng cha",
             "Nhãn cha": coarse_names[r["pred_coarse_raw"]],
             "Xác suất": f"{r['p_coarse_raw'][r['pred_coarse_raw']]:.1%}",
             "Nhất quán": "✓ có" if r["consistent"] else "✕ không"},
            {"Chế độ": "hard", "Cách làm": "tra bảng từ nhãn con",
             "Nhãn cha": coarse_names[r["pred_hard"]],
             "Xác suất": "—",
             "Nhất quán": "✓ luôn luôn"},
            {"Chế độ": "marginal", "Cách làm": "cộng xác suất nhãn con cùng nhóm",
             "Nhãn cha": coarse_names[r["pred_coarse_marg"]],
             "Xác suất": f"{r['p_coarse_marg'][r['pred_coarse_marg']]:.1%}",
             "Nhất quán": ("✓ có" if r["pred_coarse_marg"] == r["pred_hard"]
                           else "✕ không")},
        ]
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption(
            f"Mức vi phạm mềm (soft_viol) = **{r['soft_viol']:.4f}** — tổng phần "
            "xác suất nhãn con vượt quá xác suất nhãn cha tương ứng. Bằng 0 "
            "nghĩa là không vi phạm luật logic."
        )
