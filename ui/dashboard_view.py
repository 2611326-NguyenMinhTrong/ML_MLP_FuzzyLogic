"""Tab 3: bảng tổng hợp và đồ thị trade-off từ results.csv.

Các số liệu (Pareto, λ đề xuất, chênh lệch) đều được tính lại trực tiếp từ
results.csv.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.registry import RESULTS_DIR
from ui.single_view import FINE_HUE, MUTED, _mode

# Mức accuracy tối đa chấp nhận hy sinh khi chọn λ đề xuất
ACC_BUDGET = 1.0

BASE_CFG = dict(residuum="lukasiewicz", penalty="linear", warmup=3, epochs=20)
MODE_ORDER = ["raw", "hard", "marginal"]
MODE_VN = {"raw": "raw (đầu ra thô)", "hard": "hard (hậu xử lý cứng)",
          "marginal": "marginal"}

SUCCESS_TEXT = {"light": "#006300", "dark": "#0ca30c"}


@st.cache_data(show_spinner=False)
def _load(path_str, mtime):
    """Đọc results.csv (cache theo thời điểm sửa file)."""
    return pd.read_csv(path_str)


def _base(df):
    m = ((df.residuum == BASE_CFG["residuum"]) & (df.penalty == BASE_CFG["penalty"])
         & (df.warmup == BASE_CFG["warmup"]) & (df.epochs == BASE_CFG["epochs"]))
    return df[m]


def _pivot(df, metric, split):
    """rows=λ, cols=mode. Trả (bảng hiển thị 'mean ± std', bảng số để tô đậm)."""
    sub = _base(df)
    sub = sub[sub.split == split]
    g = sub.groupby(["lambda", "mode"])[metric].agg(["mean", "std", "count"])
    g = g.reset_index()
    piv_m = g.pivot(index="lambda", columns="mode", values="mean")
    piv_s = g.pivot(index="lambda", columns="mode", values="std").fillna(0.0)
    cols = [c for c in MODE_ORDER if c in piv_m.columns]
    piv_m, piv_s = piv_m[cols], piv_s[cols]
    disp = pd.DataFrame(index=piv_m.index)
    for c in cols:
        disp[MODE_VN[c]] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(piv_m[c], piv_s[c])]
    piv_m.columns = [MODE_VN[c] for c in cols]
    return disp, piv_m


def _style_best(disp, numeric, mode):
    """Bôi đậm + tô màu giá trị TỐT NHẤT (cao nhất) mỗi cột — không dùng nền màu
    để không phải lo tương phản sáng/tối, chỉ đổi màu + độ đậm của chữ."""
    color = SUCCESS_TEXT[mode]

    def hl(col):
        best_idx = numeric[col.name].idxmax()
        return [f"font-weight:700;color:{color}" if idx == best_idx else ""
                for idx in numeric.index]

    return disp.style.apply(hl, axis=0)


def _pareto_frontier(points):
    """points: list[(lambda, acc, con)]. Trả set các λ KHÔNG bị điểm nào khác
    thống trị (dominate) trên cả hai trục acc và consist (càng cao càng tốt)."""
    eff = set()
    for lam_i, acc_i, con_i in points:
        dominated = any(
            (acc_j >= acc_i and con_j >= con_i and (acc_j > acc_i or con_j > con_i))
            for lam_j, acc_j, con_j in points if lam_j != lam_i
        )
        if not dominated:
            eff.add(lam_i)
    return eff


def _pick_lambda_star(df):
    """Chọn λ đề xuất trên tập validation: trong các λ>0, lấy consistency cao
    nhất với điều kiện accuracy giảm không quá ACC_BUDGET so với B0."""
    sub = _base(df)
    sub = sub[(sub["mode"] == "raw") & (sub.split == "val")]
    if sub.empty:
        return None
    acc = sub.groupby("lambda").acc_fine.mean()
    con = sub.groupby("lambda").consist.mean()
    if 0.0 not in acc.index:
        return None
    acc0 = acc.loc[0.0]
    drop = acc0 - acc
    candidates = [lam for lam in acc.index if lam > 0 and drop.loc[lam] <= ACC_BUDGET]
    if not candidates:
        candidates = [lam for lam in acc.index if lam > 0]
        if not candidates:
            return None
    return max(candidates, key=lambda lam: con.loc[lam])


def _tradeoff_chart(df, split, lambda_star, mode):
    sub = _base(df)
    sub = sub[(sub["mode"] == "raw") & (sub.split == split)]
    g = sub.groupby("lambda").agg(
        acc_m=("acc_fine", "mean"), acc_s=("acc_fine", "std"),
        con_m=("consist", "mean"), con_s=("consist", "std")).reset_index()
    g = g.sort_values("lambda")

    points = list(zip(g["lambda"], g["acc_m"], g["con_m"]))
    pareto = _pareto_frontier(points)
    g["pareto"] = g["lambda"].isin(pareto)

    hue = FINE_HUE[mode]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["acc_m"], y=g["con_m"], mode="lines",
        line=dict(color=hue, width=2), hoverinfo="skip", showlegend=False,
    ))
    # Điểm λ* (nếu có) được vẽ riêng bằng marker hình sao ở dưới — loại nó
    # khỏi hai lớp còn lại để không chồng 2 marker + 2 nhãn tại cùng một chỗ.
    is_star = g["lambda"] == lambda_star if lambda_star is not None else g["lambda"] < 0
    g_rest = g[~is_star]
    normal = g_rest[~g_rest["pareto"]]
    fig.add_trace(go.Scatter(
        x=normal["acc_m"], y=normal["con_m"], mode="markers+text",
        marker=dict(color=hue, size=10, opacity=0.45,
                    line=dict(color=hue, width=1)),
        text=[f"λ={v:g}" for v in normal["lambda"]],
        textposition="top center", textfont=dict(color=MUTED, size=11),
        error_x=dict(type="data", array=normal["acc_s"], color=MUTED, thickness=1),
        error_y=dict(type="data", array=normal["con_s"], color=MUTED, thickness=1),
        hovertemplate="λ=%{customdata:g} (bị Pareto-dominate)<br>"
                      "acc=%{x:.2f}%, consist=%{y:.2f}%<extra></extra>",
        customdata=normal["lambda"], name="bị dominate", showlegend=False,
    ))
    pare = g_rest[g_rest["pareto"]]
    fig.add_trace(go.Scatter(
        x=pare["acc_m"], y=pare["con_m"], mode="markers+text",
        marker=dict(color=hue, size=13, line=dict(color=MUTED, width=1.5)),
        text=[f"λ={v:g}" for v in pare["lambda"]],
        textposition="top center", textfont=dict(color=MUTED, size=12),
        error_x=dict(type="data", array=pare["acc_s"], color=MUTED, thickness=1),
        error_y=dict(type="data", array=pare["con_s"], color=MUTED, thickness=1),
        hovertemplate="λ=%{customdata:g} (Pareto-hiệu quả)<br>"
                      "acc=%{x:.2f}%, consist=%{y:.2f}%<extra></extra>",
        customdata=pare["lambda"], name="Pareto-hiệu quả", showlegend=False,
    ))
    if lambda_star is not None and is_star.any():
        row = g[is_star].iloc[0]
        fig.add_trace(go.Scatter(
            x=[row["acc_m"]], y=[row["con_m"]], mode="markers+text",
            marker=dict(color=hue, size=20, symbol="star",
                        line=dict(color=MUTED, width=1.5)),
            text=[f"λ={lambda_star:g} (đề xuất)"],
            textposition="top center", textfont=dict(color=MUTED, size=12),
            error_x=dict(type="data", array=[row["acc_s"]], color=MUTED, thickness=1),
            error_y=dict(type="data", array=[row["con_s"]], color=MUTED, thickness=1),
            hovertemplate=f"λ*={lambda_star:g} (đề xuất)<br>"
                          "acc=%{x:.2f}%, consist=%{y:.2f}%<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=f"Trade-off accuracy ↔ consistency theo λ ({split}, mean±std)",
                  font=dict(size=14, color=MUTED)),
        # automargin để nhãn trục không bị cắt
        xaxis=dict(title=dict(text="Accuracy nhãn con (%)", font=dict(color=MUTED)),
                  gridcolor="rgba(137,135,129,0.18)", tickfont=dict(color=MUTED),
                  automargin=True),
        yaxis=dict(title=dict(text="Consistency (%)", font=dict(color=MUTED)),
                  gridcolor="rgba(137,135,129,0.18)", tickfont=dict(color=MUTED),
                  automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=460, margin=dict(l=70, r=30, t=56, b=60, autoexpand=True),
    )
    return fig, g


def render():
    p = RESULTS_DIR / "results.csv"
    if not p.exists():
        st.warning(
            f"Không tìm thấy `results/results.csv`.\n\n"
            f"Chép từ `train_model_colab/drive_FuzzyMLP/results/results.csv` "
            f"(nếu chung máy) hoặc tải từ Google Drive rồi đặt vào "
            f"`{p.parent}/`. Xem README mục *Chuẩn bị dữ liệu*."
        )
        return

    df = _load(str(p), p.stat().st_mtime)
    required = {"lambda", "seed", "residuum", "penalty", "warmup", "mode",
               "split", "acc_fine", "consist", "exact", "epochs"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"`results.csv` thiếu cột: {', '.join(sorted(missing))}.")
        return

    mode = _mode()
    split = st.radio("Tập dữ liệu", ["test", "val"], horizontal=True,
                     help="test = số liệu chốt cuối; val = số liệu dùng để chọn λ*")

    # Bảng pivot λ × chế độ
    st.subheader(f"Bảng tổng hợp theo λ × chế độ suy luận  ({split}, mean ± std trên các seed)",
                 anchor=False)
    c1, c2 = st.columns(2)
    for col, (label, metric) in zip((c1, c2),
                                    [("Consistency (%)", "consist"),
                                     ("Exact match — cả 2 tầng đúng (%)", "exact")]):
        disp, numeric = _pivot(df, metric, split)
        if disp.empty:
            col.info(f"Không có dữ liệu {label.lower()} cho tập {split}.")
            continue
        with col:
            st.caption(label + "  ·  đậm = giá trị cao nhất mỗi cột")
            st.dataframe(_style_best(disp, numeric, mode), width="stretch")

    acc_disp, _ = _pivot(df, "acc_fine", split)
    if not acc_disp.empty:
        # acc_fine không đổi theo chế độ nên chỉ cần một cột đại diện
        first_col = acc_disp.columns[0]
        st.caption(
            f"Accuracy nhãn con theo λ ({split}, không đổi theo chế độ vì chỉ "
            f"phụ thuộc đầu ra tầng con): " +
            " · ".join(f"λ={lam:g}: {v}" for lam, v in
                      zip(acc_disp.index, acc_disp[first_col]))
        )

    st.divider()

    # --- Đồ thị trade-off + Pareto ----------------------------------------
    st.subheader("Đường đánh đổi accuracy ↔ consistency", anchor=False)
    lambda_star = _pick_lambda_star(df)
    fig, g = _tradeoff_chart(df, split, lambda_star, mode)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    n_pareto = int(g["pareto"].sum())
    n_dom = len(g) - n_pareto
    st.caption(
        f"● đậm = **Pareto-hiệu quả** ({n_pareto}/{len(g)} cấu hình λ) — không "
        f"bị λ nào khác thắng đồng thời cả accuracy lẫn consistency. ○ mờ = "
        f"**bị Pareto-dominate** ({n_dom} cấu hình) — luôn có ít nhất một λ "
        f"khác tốt hơn hoặc bằng trên cả hai trục."
    )

    # --- Kết luận nhanh -----------------------------------------------------
    st.divider()
    st.subheader("Kết luận nhanh", anchor=False)

    if n_dom > 0:
        dom_list = ", ".join(f"λ={v:g}" for v in sorted(g[~g["pareto"]]["lambda"]))
        st.markdown(
            f"- Trên **{split}**, {n_dom} cấu hình ({dom_list}) bị Pareto-dominate — "
            f"loại khỏi cân nhắc vì luôn có lựa chọn khác tốt hơn hoặc bằng."
        )
    else:
        st.markdown(
            f"- Trên **{split}**, cả {len(g)} giá trị λ đều Pareto-hiệu quả — "
            "accuracy và consistency đánh đổi đơn điệu, không có λ nào bị loại "
            "thuần túy bởi so sánh trội."
        )

    if lambda_star is None:
        st.markdown("- Không đủ dữ liệu trên **val** để tính λ\\* theo quy tắc ngân sách.")
    else:
        st.markdown(
            f"- **λ\\* đề xuất = {lambda_star:g}** (quy tắc: consistency cao nhất "
            f"trong các λ>0 mà accuracy giảm ≤ {ACC_BUDGET:g} điểm so với B0, "
            f"chọn trên **validation** — không bao giờ trên test)."
        )
        row = g[g["lambda"] == lambda_star]
        if not row.empty:
            row = row.iloc[0]
            is_pareto = bool(row["pareto"])
            st.markdown(
                f"  - Trên **{split}**, λ\\*={lambda_star:g} "
                f"{'**là**' if is_pareto else '**KHÔNG phải**'} một điểm Pareto-hiệu quả."
            )

        piv_disp, piv_num = _pivot(df, "consist", split)
        star_row = piv_disp.loc[lambda_star] if lambda_star in piv_disp.index else None
        if star_row is not None:
            raw_c = MODE_VN["raw"]
            hard_c = MODE_VN.get("hard")
            marg_c = MODE_VN.get("marginal")
            bits = [f"chế độ **raw** = {star_row[raw_c]}%"]
            if hard_c in piv_num.columns:
                gap_h = piv_num.loc[lambda_star, hard_c] - piv_num.loc[lambda_star, raw_c]
                bits.append(f"**hard** = {star_row[hard_c]}% (raw kém hơn "
                           f"{gap_h:.2f} điểm)")
            if marg_c in piv_num.columns:
                gap_m = piv_num.loc[lambda_star, marg_c] - piv_num.loc[lambda_star, raw_c]
                sign = "kém hơn" if gap_m > 0 else "nhỉnh hơn"
                bits.append(f"**marginal** = {star_row[marg_c]}% (raw {sign} "
                           f"{abs(gap_m):.2f} điểm)")
            st.markdown(
                f"- Tại λ\\*={lambda_star:g} trên **{split}**, consistency của "
                + "; ".join(bits) + ". Vì `hard` luôn đạt 100% *theo cách xây dựng* "
                "và không tốn công huấn luyện thêm, khoảng cách với `raw` chính là "
                "cái giá phải trả để có một đầu ra tầng cha \"tự nhiên\" thay vì "
                "hậu xử lý."
            )

    st.caption(
        "Các số liệu ở tab này được tính trực tiếp từ `results.csv`. "
        "Phân tích đầy đủ (McNemar, nhóm chứng, V/NV) nằm trong báo cáo."
    )
