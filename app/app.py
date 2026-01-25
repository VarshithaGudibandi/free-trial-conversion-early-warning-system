import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# Repo root (works locally + Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# Page config + CSS
# =========================
st.set_page_config(
    page_title="Free-Trial to Paid Conversion Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
      }
      div[data-testid="stDataFrame"] { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Helpers
# =========================
def conversion_by_bin(df, col, bins, labels=None):
    tmp = df.copy()
    tmp["bin"] = pd.cut(tmp[col], bins=bins, labels=labels, include_lowest=True)
    out = (
        tmp.groupby("bin")
        .agg(users=("user_id", "count"), converters=("converted_to_paid", "sum"))
        .reset_index()
    )
    out["conversion_rate"] = (out["converters"] / out["users"]).fillna(0)
    return out


@st.cache_data
def load_model():
    model_path = BASE_DIR / "models" / "rf_conversion_model_final.pkl"
    cols_path = BASE_DIR / "models" / "rf_feature_cols_final.pkl"
    return joblib.load(model_path), joblib.load(cols_path)


@st.cache_data
def load_dataset():
    path = BASE_DIR / "data" / "raw" / "customer_conversion_model_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_importance():
    path = BASE_DIR / "models" / "feature_importance_random_forest.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# =========================
# Load artifacts
# =========================
rf, feature_cols = load_model()
df_model = load_dataset()
fi_df = load_feature_importance()

dataset_conv_rate = (
    df_model["converted_to_paid"].mean()
    if df_model is not None and "converted_to_paid" in df_model.columns
    else None
)

# =========================
# Header
# =========================
st.markdown(
    """
    <h1 style="font-size:42px;font-weight:700;margin-bottom:6px;">
        Free-Trial to Paid Conversion Explorer
    </h1>
    <p style="font-size:16px;color:#6b7280;margin-top:0;">
        Interactive early-warning system to estimate conversion probability and support targeted product interventions.
    </p>
    """,
    unsafe_allow_html=True
)

# =========================
# Core inputs on page (NOT sidebar)
# =========================
st.markdown("### Core trial behavior inputs")

c1, c2, c3 = st.columns(3)

with c1:
    time_spent_min = st.slider("Time spent on site (minutes)", 0, 300, 30)
    pages_viewed = st.slider("Pages viewed", 0, 100, 10)

with c2:
    form_submissions = st.slider("Form submissions", 0, 10, 1)
    downloads = st.slider("Downloads", 0, 20, 0)

with c3:
    email_intensity = st.slider("Total marketing emails", 0, 40, 4)
    response_time_hours = st.slider("Response time (hours)", 0.0, 48.0, 4.0)

st.markdown("### Secondary signals")

c4, c5, c6 = st.columns(3)

with c4:
    age = st.slider("Age", 18, 80, 30)

with c5:
    social_media_engagement = st.slider("Social media engagement score", 0, 100, 10)

with c6:
    ctr_product_page = st.slider("Product page CTR (%)", 0.0, 100.0, 5.0)

# =========================
# Advanced inputs in sidebar (collapsed by default)
# =========================
st.sidebar.header("Advanced inputs (optional)")

lead_status = st.sidebar.selectbox("Lead status", ["Cold", "Warm", "Hot"])
payment_history = st.sidebar.selectbox("Payment history", ["No Payment", "Good"])
lead_source = st.sidebar.selectbox(
    "Lead source", ["Organic", "Email", "Referral", "Social Media"]
)
device_type = st.sidebar.selectbox("Device type", ["Desktop", "Mobile", "Tablet"])

# =========================
# Feature prep + prediction
# =========================
lead_status_map = {"Cold": 0, "Warm": 1, "Hot": 2}
payment_map = {"Good": 1, "No Payment": 0}

lead_status_score = lead_status_map[lead_status]
payment_history_good = payment_map[payment_history]

engagement_score = (
    time_spent_min
    + pages_viewed * 2
    + form_submissions * 3
    + downloads * 2
    + social_media_engagement
)

data = {
    "age": age,
    "lead_status_score": lead_status_score,
    "payment_history_good": payment_history_good,
    "time_spent_min": time_spent_min,
    "pages_viewed": pages_viewed,
    "email_intensity": email_intensity,
    "form_submissions": form_submissions,
    "downloads": downloads,
    "ctr_product_page": ctr_product_page,
    "response_time_hours": response_time_hours,
    "social_media_engagement": social_media_engagement,
    "engagement_score": engagement_score,
    "is_mobile": int(device_type == "Mobile"),
    "is_tablet": int(device_type == "Tablet"),
    "is_desktop": int(device_type == "Desktop"),
    "is_organic_lead": int(lead_source == "Organic"),
    "is_email_lead": int(lead_source == "Email"),
    "is_referral_lead": int(lead_source == "Referral"),
    "is_social_lead": int(lead_source == "Social Media"),
}

x = pd.DataFrame([data])[feature_cols]
prob = float(rf.predict_proba(x)[0, 1])
prediction = prob * 100

# =========================
# Hero row
# =========================
st.markdown("---")
h1, h2 = st.columns([2, 3])

with h1:
    st.markdown(
        f"""
        <div style="background:#f9fafb;padding:22px;border-radius:12px;border:1px solid #e5e7eb;">
            <p style="font-size:14px;color:#6b7280;margin-bottom:6px;">
                Predicted conversion probability
            </p>
            <p style="font-size:40px;font-weight:800;margin:0;color:#111827">
                {prediction:.2f}%
            </p>
            <p style="font-size:13px;color:#9ca3af;margin-top:10px;">
                Likelihood this trial user converts to a paid plan
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with h2:
    st.markdown(
        """
        <div style="
            padding:22px;
            border-radius:12px;
            border:1px solid #e5e7eb;
            background:#ffffff;
        ">
            <p style="font-size:14px;font-weight:800;color:#111827;margin:0;">
                Risk interpretation
            </p>

            <ul style="
                font-size:13px;
                line-height:1.8;
                margin-top:10px;
                color:#111827;
            ">
                <li><b>High risk:</b> low engagement & weak activation</li>
                <li><b>Medium risk:</b> intent present, needs guidance</li>
                <li><b>Low risk:</b> strong activation & consistent use</li>
            </ul>

            <p style="font-size:13px;color:#4b5563;margin-top:10px;">
                Use this output to trigger targeted onboarding, feature education, and lifecycle messaging.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Tabs
# =========================
st.markdown("<br>", unsafe_allow_html=True)
tab_predict, tab_insights, tab_benchmarks = st.tabs(
    ["🔮 Predict a User", "📊 Model Insights", "🏷 Industry Benchmarks"]
)

# =========================
# TAB 1: Predict
# =========================
with tab_predict:
    st.subheader("Prediction result")
    st.write(f"**{prob:.2%}** chance this user converts to paid.")

    with st.expander("View feature values used for prediction"):
        st.dataframe(pd.DataFrame([data]), use_container_width=True)

# =========================
# TAB 2: Insights
# =========================
with tab_insights:
    st.subheader("Model & Dataset Insights")

    if df_model is None:
        st.warning("Dataset file not found in data/raw/.")
    else:
        k1, k2, k3 = st.columns(3)

        converters = df_model["converted_to_paid"] == 1
        non_converters = df_model["converted_to_paid"] == 0

        avg_eng_conv = df_model.loc[converters, "engagement_score"].mean()
        avg_eng_non = df_model.loc[non_converters, "engagement_score"].mean()

        avg_resp_conv = df_model.loc[converters, "response_time_hours"].mean()
        avg_resp_non = df_model.loc[non_converters, "response_time_hours"].mean()

        with k1:
            st.metric("Overall conversion rate", f"{dataset_conv_rate:.2%}" if dataset_conv_rate is not None else "N/A")

        with k2:
            st.metric(
                "Avg engagement (converters)",
                f"{avg_eng_conv:.1f}",
                f"{(avg_eng_conv - avg_eng_non):.1f} vs non-converters",
            )

        with k3:
            st.metric(
                "Avg response time (converters)",
                f"{avg_resp_conv:.1f} h",
                f"{(avg_resp_conv - avg_resp_non):.1f} h vs non-converters",
            )

        st.markdown("---")

        bins_pages = [0, 5, 10, 20, 50, np.inf]
        labels_pages = ["0–5", "6–10", "11–20", "21–50", "50+"]

        bins_eng = [0, 50, 100, 200, np.inf]
        labels_eng = ["Very low", "Low", "Medium", "High"]

        bins_email = [0, 1, 3, 6, 10, np.inf]
        labels_email = ["0", "1–2", "3–5", "6–9", "10+"]

        bins_resp = [0, 1, 4, 12, 24, np.inf]
        labels_resp = ["≤1h", "1–4h", "4–12h", "12–24h", ">24h"]

        conv_vs_pages = conversion_by_bin(df_model, "pages_viewed", bins_pages, labels_pages)
        conv_vs_eng = conversion_by_bin(df_model, "engagement_score", bins_eng, labels_eng)
        conv_vs_email = conversion_by_bin(df_model, "email_intensity", bins_email, labels_email)
        conv_vs_resp = conversion_by_bin(df_model, "response_time_hours", bins_resp, labels_resp)

        a, b = st.columns(2)
        with a:
            st.markdown("### Conversion by pages viewed")
            st.bar_chart(conv_vs_pages.set_index("bin")["conversion_rate"])
            with st.expander("Underlying numbers: pages viewed"):
                st.dataframe(conv_vs_pages, use_container_width=True)

        with b:
            st.markdown("### Conversion by engagement score")
            st.bar_chart(conv_vs_eng.set_index("bin")["conversion_rate"])
            with st.expander("Underlying numbers: engagement"):
                st.dataframe(conv_vs_eng, use_container_width=True)

        c, d = st.columns(2)
        with c:
            st.markdown("### Conversion by email intensity")
            st.bar_chart(conv_vs_email.set_index("bin")["conversion_rate"])
            with st.expander("Underlying numbers: email intensity"):
                st.dataframe(conv_vs_email, use_container_width=True)

        with d:
            st.markdown("### Conversion by response time")
            st.bar_chart(conv_vs_resp.set_index("bin")["conversion_rate"])
            with st.expander("Underlying numbers: response time"):
                st.dataframe(conv_vs_resp, use_container_width=True)

        st.markdown("---")
        if fi_df is not None and {"feature", "importance"}.issubset(fi_df.columns):
            st.markdown("### Feature importance (Random Forest)")
            st.bar_chart(fi_df.set_index("feature")["importance"])
        else:
            st.info("Feature importance file missing or invalid.")

# =========================
# TAB 3: Benchmarks
# =========================
with tab_benchmarks:
    st.subheader("Industry Benchmarks (trial-to-paid)")

    bench = pd.DataFrame(
        {
            "scenario": [
                "This dataset",
                "Opt-in free trial (B2B SaaS)",
                "Opt-out free trial (B2B SaaS)",
            ],
            "conversion_rate": [
                dataset_conv_rate if dataset_conv_rate is not None else np.nan,
                0.18,
                0.49,
            ],
            "source": [
                "Kaggle synthetic",
                "Public SaaS benchmarks (opt-in)",
                "Public SaaS benchmarks (opt-out)",
            ],
        }
    )
    bench["conversion_rate (%)"] = (bench["conversion_rate"] * 100).round(1)

    st.dataframe(bench[["scenario", "conversion_rate (%)", "source"]], use_container_width=True)

    if dataset_conv_rate is not None:
        delta_vs_optin = (dataset_conv_rate - 0.18) * 100
        st.metric(
            "Gap vs typical B2B opt-in benchmark (~18%)",
            f"{dataset_conv_rate:.2%}",
            f"{delta_vs_optin:.1f} pp",
        )

    st.caption(
        "Benchmarks are approximate and based on public SaaS trial-to-paid studies. "
        "Real products will vary by segment, pricing, and trial design."
    )