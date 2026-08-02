"""Ứng dụng Streamlit: chọn checkpoint ở sidebar và hiển thị 3 tab.

Logic mô hình nằm trong core/, giao diện từng tab nằm trong ui/.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.inference import CheckpointError, load_ckpt
from core.registry import list_checkpoints, list_samples
from ui import compare_view, dashboard_view, single_view

st.set_page_config(page_title="MLP + Fuzzy Logic — CIFAR-100",
                   page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Đang nạp checkpoint...")
def get_model(path_str):
    """Cache theo đường dẫn để đổi tab không phải nạp lại mô hình."""
    return load_ckpt(Path(path_str))


st.title("MLP + Fuzzy Logic — phân loại ảnh phân cấp")

items = list_checkpoints()
ok_items = [d for d in items if not d.get("error")]

if not ok_items:
    st.error(
        "**Chưa có checkpoint nào trong `checkpoints/`.**\n\n"
        "App cần ít nhất một file `.pt` do notebook sinh ra. "
        "Xem mục *Chuẩn bị dữ liệu* trong `README.md`."
    )
    st.stop()

with st.sidebar:
    st.header("Mô hình")
    labels = [d["label"] for d in ok_items]
    pick = st.selectbox("Chọn checkpoint", range(len(ok_items)),
                        format_func=lambda i: labels[i])
    chosen = ok_items[pick]

    try:
        model, meta = get_model(str(chosen["path"]))
    except CheckpointError as e:
        st.error(str(e))
        st.stop()

    vm = chosen.get("val_metrics") or {}
    if vm:
        st.metric("Accuracy nhãn con (val)", f"{vm.get('acc_f', 0):.2f}%")
        st.metric("Tính nhất quán (val)", f"{vm.get('consist', 0):.2f}%")

    ti_bits = [f"λ = {chosen['lambda']:g}", f"seed {chosen['seed']}",
               f"{chosen['epochs']} epoch", f"residuum {chosen['residuum']}"]
    st.caption(" · ".join(str(b) for b in ti_bits))

    bad = [d for d in items if d.get("error")]
    if bad:
        with st.expander(f"⚠️ {len(bad)} checkpoint lỗi (đã bỏ qua)"):
            for d in bad:
                st.caption(f"**{d['name']}** — {d['error']}")

    st.divider()
    st.caption(f"{len(list_samples())} ảnh mẫu · "
               f"{meta['config']['n_fine']} nhãn con → "
               f"{meta['config']['n_coarse']} nhãn cha")

tab1, tab2, tab3 = st.tabs(["Một ảnh", "So sánh B0 vs M", "Tổng quan"])

with tab1:
    single_view.render(model, meta)

with tab2:
    compare_view.render()

with tab3:
    dashboard_view.render()
