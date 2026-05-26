import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Anomaly Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-val  { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
  .metric-label{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
  .alert-box   {
    background: #450a0a;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 13px;
    color: #fca5a5;
  }
  .stable-box  {
    background: #052e16;
    border: 1px solid #22c55e;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 13px;
    color: #86efac;
  }
</style>
""", unsafe_allow_html=True)
 
# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    results   = pd.read_csv("results_df.csv")
    patients  = pd.read_csv("patient_results.csv")
    vitals_df = pd.read_csv("vitals_wide_clean_final.csv")
    vitals_df["charttime"] = pd.to_datetime(vitals_df["charttime"])
    return results, patients, vitals_df
 
results_df, patient_results, vitals_df = load_data()
 
VITALS    = ["HR", "NBPd", "NBPs", "RR", "SpO2"]
THRESHOLD = results_df[results_df["anomaly"] == True]["error"].min()
 
# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/hospital.png", width=60
)
st.sidebar.title("ICU AI Monitor")
st.sidebar.caption("LSTM Autoencoder — Anomaly Detection")
st.sidebar.divider()
 
view = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔴 Anomaly Alerts", "📈 Patient Vitals", "📊 Model Insights"],
)
 
st.sidebar.divider()
threshold_pct = st.sidebar.slider(
    "Anomaly threshold percentile",
    min_value=90, max_value=99, value=95, step=1,
    help="Raise to reduce false alarms. Lower to catch more anomalies."
)
dynamic_threshold = float(
    np.percentile(results_df["error"], threshold_pct)
)
results_df["anomaly_dynamic"] = results_df["error"] > dynamic_threshold
 
patient_results_dynamic = (
    results_df.groupby("stay_id")
    .agg(
        max_error    = ("error",            "max"),
        any_anomaly  = ("anomaly_dynamic",  "any"),
        n_windows    = ("error",            "count"),
        n_anomalies  = ("anomaly_dynamic",  "sum"),
    )
    .reset_index()
)
 
def risk_label(row):
    if row["any_anomaly"] and row["max_error"] > dynamic_threshold * 1.5:
        return "🔴 Critical"
    elif row["any_anomaly"]:
        return "🟡 Elevated"
    else:
        return "🟢 Stable"
 
patient_results_dynamic["risk"] = patient_results_dynamic.apply(
    risk_label, axis=1
)
 
# ════════════════════════════════════════════════════════════
#  VIEW 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if view == "🏠 Overview":
    st.title("🏥 ICU Real-Time Anomaly Monitoring")
    st.caption(
        "LSTM Autoencoder trained on MIMIC-IV demo v2.2 · "
        "5 vitals · 24-hour sliding windows"
    )
    st.divider()
 
    # ── Top metrics ──
    total    = len(patient_results_dynamic)
    critical = (patient_results_dynamic["risk"] == "🔴 Critical").sum()
    elevated = (patient_results_dynamic["risk"] == "🟡 Elevated").sum()
    stable   = (patient_results_dynamic["risk"] == "🟢 Stable").sum()
    anom_pct = round(results_df["anomaly_dynamic"].mean() * 100, 1)
 
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients",   total)
    c2.metric("🔴 Critical",      critical)
    c3.metric("🟡 Elevated",      elevated)
    c4.metric("🟢 Stable",        stable)
    c5.metric("Anomaly Rate",     f"{anom_pct}%")
 
    st.divider()
 
    col_left, col_right = st.columns([1.6, 1])
 
    # ── Bar chart ──
    with col_left:
        st.subheader("Max Anomaly Score per Patient")
        bar_df = patient_results_dynamic.sort_values(
            "max_error", ascending=False
        )
        colors = bar_df["risk"].map({
            "🔴 Critical": "#ef4444",
            "🟡 Elevated": "#f59e0b",
            "🟢 Stable":   "#22c55e",
        })
        fig = go.Figure(go.Bar(
            x=bar_df["stay_id"].astype(str),
            y=bar_df["max_error"],
            marker_color=colors,
            hovertemplate=(
                "Stay: %{x}<br>"
                "Max Error: %{y:.5f}<br>"
                "<extra></extra>"
            )
        ))
        fig.add_hline(
            y=dynamic_threshold,
            line_dash="dash", line_color="white",
            annotation_text=f"Threshold ({threshold_pct}th pct)",
            annotation_position="top right"
        )
        fig.update_layout(
            xaxis_title="Stay ID",
            yaxis_title="Max Reconstruction Error",
            xaxis=dict(showticklabels=False),
            height=350,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Risk donut ──
    with col_right:
        st.subheader("Risk Distribution")
        fig2 = go.Figure(go.Pie(
            labels=["Critical", "Elevated", "Stable"],
            values=[critical, elevated, stable],
            hole=0.55,
            marker_colors=["#ef4444", "#f59e0b", "#22c55e"],
            textinfo="label+percent",
        ))
        fig2.update_layout(
            height=350,
            showlegend=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Error distribution ──
    st.subheader("Reconstruction Error Distribution")
    fig3 = px.histogram(
        results_df, x="error", nbins=80,
        color="anomaly_dynamic",
        color_discrete_map={True: "#ef4444", False: "#3b82f6"},
        labels={"error": "Reconstruction Error",
                "anomaly_dynamic": "Anomaly"},
        height=280,
    )
    fig3.add_vline(
        x=dynamic_threshold, line_dash="dash",
        line_color="white",
        annotation_text="Threshold",
        annotation_position="top right"
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)
 
# ════════════════════════════════════════════════════════════
#  VIEW 2 — ANOMALY ALERTS
# ════════════════════════════════════════════════════════════
elif view == "🔴 Anomaly Alerts":
    st.title("🔴 Anomaly Alerts")
    st.caption("Patients whose vital sign patterns deviated from learned normal baseline.")
    st.divider()
 
    alerted = patient_results_dynamic[
        patient_results_dynamic["any_anomaly"]
    ].sort_values("max_error", ascending=False)
 
    st.markdown(f"**{len(alerted)} patients triggered anomaly alerts** "
                f"at the {threshold_pct}th percentile threshold.")
    st.divider()
 
    for _, row in alerted.iterrows():
        risk_color = {"🔴 Critical": "alert-box",
                      "🟡 Elevated": "alert-box",
                      "🟢 Stable":   "stable-box"}.get(row["risk"], "alert-box")
        st.markdown(
            f'<div class="{risk_color}">'
            f'<b>{row["risk"]}  —  Stay {row["stay_id"]}</b><br>'
            f'Max error: <b>{row["max_error"]:.5f}</b>  |  '
            f'Anomaly windows: <b>{int(row["n_anomalies"])}</b> / {int(row["n_windows"])}  |  '
            f'Anomaly rate: <b>{row["n_anomalies"]/row["n_windows"]*100:.1f}%</b>'
            f'</div>',
            unsafe_allow_html=True
        )
 
    st.divider()
    st.subheader("Anomaly Windows Over Time")
    stay_options = alerted["stay_id"].tolist()
    selected_alert = st.selectbox("Select alerted patient:", stay_options)
 
    stay_errors = results_df[results_df["stay_id"] == selected_alert].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=stay_errors["error"],
        mode="lines+markers",
        name="Reconstruction error",
        line=dict(color="#3b82f6", width=1.5),
        marker=dict(
            color=["#ef4444" if a else "#3b82f6"
                   for a in stay_errors["anomaly_dynamic"]],
            size=6,
        )
    ))
    fig.add_hline(
        y=dynamic_threshold, line_dash="dash",
        line_color="white",
        annotation_text="Threshold"
    )
    fig.update_layout(
        title=f"Reconstruction Error Over Time — Stay {selected_alert}",
        xaxis_title="Window index (1 per hour)",
        yaxis_title="Reconstruction Error",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
 
# ════════════════════════════════════════════════════════════
#  VIEW 3 — PATIENT VITALS
# ════════════════════════════════════════════════════════════
elif view == "📈 Patient Vitals":
    st.title("📈 Patient Vital Sign Trends")
    st.caption("Select a patient to view their vital sign timeline.")
    st.divider()
 
    col1, col2 = st.columns([2, 1])
    with col1:
        sorted_stays = (
            patient_results_dynamic
            .sort_values("max_error", ascending=False)["stay_id"]
            .tolist()
        )
        selected_stay = st.selectbox("Select patient stay:", sorted_stays)
    with col2:
        vital_select = st.multiselect(
            "Vitals to display:",
            VITALS, default=VITALS
        )
 
    patient_row = patient_results_dynamic[
        patient_results_dynamic["stay_id"] == selected_stay
    ].iloc[0]
 
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Status",     patient_row["risk"])
    r2.metric("Max Error",       f"{patient_row['max_error']:.5f}")
    r3.metric("Anomaly Windows", f"{int(patient_row['n_anomalies'])} / {int(patient_row['n_windows'])}")
 
    st.divider()
 
    pv = vitals_df[vitals_df["stay_id"] == selected_stay].copy()
 
    if len(vital_select) == 0:
        st.warning("Please select at least one vital sign.")
    else:
        colors = {
            "HR":   "#3b82f6",
            "NBPd": "#ef4444",
            "NBPs": "#f59e0b",
            "RR":   "#10b981",
            "SpO2": "#8b5cf6",
        }
        fig = go.Figure()
        for v in vital_select:
            if v in pv.columns:
                fig.add_trace(go.Scatter(
                    x=pv["charttime"], y=pv[v],
                    name=v,
                    line=dict(color=colors.get(v, "#fff"), width=1.5),
                    mode="lines"
                ))
        fig.update_layout(
            title=f"Vital Signs — Stay {selected_stay}  |  {patient_row['risk']}",
            xaxis_title="Time",
            yaxis_title="Value",
            height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # Error timeline for this patient
    st.subheader("Anomaly Score Timeline")
    stay_err = results_df[
        results_df["stay_id"] == selected_stay
    ].reset_index(drop=True)
 
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=stay_err["error"],
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=1.5),
        name="Error",
        marker=dict(
            color=["#ef4444" if a else "#3b82f6"
                   for a in stay_err["anomaly_dynamic"]],
            size=5,
        ),
        mode="lines+markers"
    ))
    fig2.add_hline(
        y=dynamic_threshold, line_dash="dash",
        line_color="white", annotation_text="Threshold"
    )
    fig2.update_layout(
        xaxis_title="Window (1 per hour)",
        yaxis_title="Reconstruction Error",
        height=280,
        margin=dict(t=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
 
# ════════════════════════════════════════════════════════════
#  VIEW 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════
elif view == "📊 Model Insights":
    st.title("📊 Model Insights")
    st.caption("LSTM Autoencoder performance and configuration summary.")
    st.divider()
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture",    "LSTM Autoencoder")
    c2.metric("Latent Dim",      "32")
    c3.metric("Window Size",     "24 hours")
    c4.metric("Input Features",  "5 vitals")
 
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Training Epochs", "50")
    c6.metric("Loss Function",   "MSE")
    c7.metric("Total Windows",   f"{len(results_df):,}")
    c8.metric("Threshold",       f"{dynamic_threshold:.5f}")
 
    st.divider()
 
    col1, col2 = st.columns(2)
 
    # Error box plot per vital (approximate from results)
    with col1:
        st.subheader("Error Distribution by Risk Category")
        merged = results_df.merge(
            patient_results_dynamic[["stay_id","risk"]],
            on="stay_id", how="left"
        )
        fig = px.box(
            merged, x="risk", y="error",
            color="risk",
            color_discrete_map={
                "🔴 Critical": "#ef4444",
                "🟡 Elevated": "#f59e0b",
                "🟢 Stable":   "#22c55e",
            },
            labels={"error": "Reconstruction Error", "risk": "Risk Category"},
            height=350,
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
 
    # Windows per patient
    with col2:
        st.subheader("Anomaly Windows per Patient")
        top_df = (patient_results_dynamic
                  .sort_values("n_anomalies", ascending=False)
                  .head(20))
        fig2 = px.bar(
            top_df,
            x="stay_id", y="n_anomalies",
            color="risk",
            color_discrete_map={
                "🔴 Critical": "#ef4444",
                "🟡 Elevated": "#f59e0b",
                "🟢 Stable":   "#22c55e",
            },
            labels={"n_anomalies": "Anomaly Windows",
                    "stay_id": "Stay ID"},
            height=350,
        )
        fig2.update_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False),
            margin=dict(t=10)
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    st.divider()
    st.subheader("Full Patient Summary Table")
    st.dataframe(
        patient_results_dynamic[[
            "stay_id","risk","max_error",
            "n_windows","n_anomalies"
        ]].rename(columns={
            "stay_id":     "Stay ID",
            "risk":        "Risk Status",
            "max_error":   "Max Error",
            "n_windows":   "Total Windows",
            "n_anomalies": "Anomaly Windows",
        }).sort_values("Max Error", ascending=False),
        use_container_width=True,
        height=400,
    )
 
    st.caption(
        "Dataset: MIMIC-IV Clinical Database Demo v2.2 · "
        "Model: LSTM Autoencoder · "
        "Project: Strategic Implementation of AI in Healthcare"
    )