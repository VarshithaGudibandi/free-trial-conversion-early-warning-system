import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -------------------
# Helpers
# -------------------

def conversion_by_bin(df, col, bins, labels=None):
    tmp = df.copy()
    tmp["bin"] = pd.cut(tmp[col], bins=bins, labels=labels, include_lowest=True)
    result = (
        tmp.groupby("bin")
        .agg(
            users=("user_id", "count"),
            converters=("converted_to_paid", "sum"),
        )
        .reset_index()
    )
    result["conversion_rate"] = result["converters"] / result["users"]
    return result


@st.cache_data
def load_model():
    model = joblib.load("rf_conversion_model_final.pkl")
    feature_cols = joblib.load("rf_feature_cols_final.pkl")
    return model, feature_cols


@st.cache_data
def load_model_dataset():
    try:
        df = pd.read_csv("customer_conversion_model_dataset.csv")
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_feature_importance():
    try:
        fi = pd.read_csv("feature_importance_random_forest.csv")
        return fi
    except FileNotFoundError:
        return None


# -------------------
# Load artefacts
# -------------------

rf, feature_cols = load_model()
df_model = load_model_dataset()
fi_df = load_feature_importance()

if df_model is not None:
    dataset_conv_rate = df_model["converted_to_paid"].mean()
else:
    dataset_conv_rate = None

# -------------------
# Layout
# -------------------

st.title("Free-Trial to Paid Conversion Explorer")

tab_predict, tab_insights, tab_benchmarks = st.tabs(
    ["Predict a User", "Model Insights", "Industry Benchmarks"]
)

# -------------------
# TAB 1: Predict a User
# -------------------
with tab_predict:
    st.write(
        "Use the controls on the left to describe a trial user's behaviour. "
        "The model will estimate the probability that this user converts to a paid plan."
    )

    # Sidebar controls
    age = st.sidebar.slider("Age", 18, 80, 30)
    time_spent_min = st.sidebar.slider("Time spent on site (minutes)", 0, 300, 30)
    pages_viewed = st.sidebar.slider("Pages viewed", 0, 100, 10)

    form_submissions = st.sidebar.slider("Form submissions", 0, 10, 1)
    downloads = st.sidebar.slider("Downloads", 0, 20, 0)
    social_media_engagement = st.sidebar.slider(
        "Social media engagement score", 0, 100, 10
    )

    ctr_product_page = st.sidebar.slider("Product page CTR (%)", 0.0, 100.0, 5.0)
    response_time_hours = st.sidebar.slider("Response time (hours)", 0.0, 48.0, 4.0)

    email_intensity = st.sidebar.slider(
        "Total marketing emails (initial + follow-ups)", 0, 40, 4
    )

    lead_status = st.sidebar.selectbox("Lead status", ["Cold", "Warm", "Hot"])
    payment_history = st.sidebar.selectbox("Payment history", ["No Payment", "Good"])
    lead_source = st.sidebar.selectbox(
        "Lead source", ["Organic", "Email", "Referral", "Social Media"]
    )
    device_type = st.sidebar.selectbox(
        "Device type", ["Desktop", "Mobile", "Tablet"]
    )

    # Map categoricals -> numeric
    lead_status_map = {"Cold": 0, "Warm": 1, "Hot": 2}
    payment_map = {"Good": 1, "No Payment": 0}

    lead_status_score = lead_status_map[lead_status]
    payment_history_good = payment_map[payment_history]

    is_mobile = int(device_type == "Mobile")
    is_tablet = int(device_type == "Tablet")
    is_desktop = int(device_type == "Desktop")

    is_organic_lead = int(lead_source == "Organic")
    is_email_lead = int(lead_source == "Email")
    is_referral_lead = int(lead_source == "Referral")
    is_social_lead = int(lead_source == "Social Media")

    # Engagement score – same formula as training
    engagement_score = (
        time_spent_min
        + pages_viewed * 2
        + form_submissions * 3
        + downloads * 2
        + social_media_engagement
    )

    # Build feature row for prediction
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
        "is_mobile": is_mobile,
        "is_tablet": is_tablet,
        "is_desktop": is_desktop,
        "is_organic_lead": is_organic_lead,
        "is_email_lead": is_email_lead,
        "is_referral_lead": is_referral_lead,
        "is_social_lead": is_social_lead,
    }

    x = pd.DataFrame([data])[feature_cols]
    prob = rf.predict_proba(x)[0, 1]

    st.subheader("Predicted conversion probability")
    st.write(f"**{prob:.2%}** chance this user converts to paid.")

# -------------------
# TAB 2: Model Insights (Dashboard)
# -------------------
with tab_insights:
    st.subheader("Model & Dataset Insights")

    if df_model is None:
        st.warning(
            "Dataset file not found. Place 'customer_conversion_model_dataset.csv' next to app.py."
        )
    else:
        # --- Explain formulas ---
        with st.expander("How are conversion rate and engagement score defined?"):
            st.write("""
            **Conversion rate**
            Conversion Rate = Converted Users/Total Users

            **Engagement score** (weighted sum of key trial behaviours):

            Engagement Score = time\_spent\_min + 2 * pages\_viewed+ 3 * form\_submissions + 
            2 * downloads + social\_media\_engagement
            """)

        # --- Topline KPIs ---
        converters = df_model["converted_to_paid"] == 1
        non_converters = df_model["converted_to_paid"] == 0

        avg_eng_conv = df_model.loc[converters, "engagement_score"].mean()
        avg_eng_non = df_model.loc[non_converters, "engagement_score"].mean()

        avg_resp_conv = df_model.loc[converters, "response_time_hours"].mean()
        avg_resp_non = df_model.loc[non_converters, "response_time_hours"].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Overall conversion rate",
                f"{dataset_conv_rate:.2%}",
            )
        with col2:
            st.metric(
                "Avg engagement (converters)",
                f"{avg_eng_conv:.1f}",
                f"{(avg_eng_conv - avg_eng_non):.1f} vs non-converters",
            )
        with col3:
            st.metric(
                "Avg response time (converters)",
                f"{avg_resp_conv:.1f} h",
                f"{(avg_resp_conv - avg_resp_non):.1f} h vs non-converters",
            )

        st.markdown("---")

        # -------- Top 6 feature dashboards --------

        # 1. Conversion by pages_viewed
        st.markdown("### Conversion by pages viewed")
        bins_pages = [0, 5, 10, 20, 50, np.inf]
        labels_pages = ["0–5", "6–10", "11–20", "21–50", "50+"]

        conv_vs_pages = conversion_by_bin(
            df_model, "pages_viewed", bins_pages, labels_pages
        )
        st.bar_chart(conv_vs_pages.set_index("bin")["conversion_rate"])
        with st.expander("See underlying numbers for pages viewed"):
            st.dataframe(
                conv_vs_pages.rename(columns={"bin": "pages_viewed_bucket"})
            )

        # 2. Conversion by engagement_score
        st.markdown("### Conversion by engagement score")
        bins_eng = [0, 50, 100, 200, np.inf]
        labels_eng = ["Very low", "Low", "Medium", "High"]

        conv_vs_eng = conversion_by_bin(
            df_model, "engagement_score", bins_eng, labels_eng
        )
        st.bar_chart(conv_vs_eng.set_index("bin")["conversion_rate"])
        with st.expander("See underlying numbers for engagement score"):
            st.dataframe(
                conv_vs_eng.rename(columns={"bin": "engagement_bucket"})
            )

        # 3. Conversion by email_intensity
        st.markdown("### Conversion by email intensity")
        bins_email = [0, 1, 3, 6, 10, np.inf]
        labels_email = ["0", "1–2", "3–5", "6–9", "10+"]

        conv_vs_email = conversion_by_bin(
            df_model, "email_intensity", bins_email, labels_email
        )
        st.bar_chart(conv_vs_email.set_index("bin")["conversion_rate"])
        with st.expander("See underlying numbers for email intensity"):
            st.dataframe(
                conv_vs_email.rename(columns={"bin": "email_intensity_bucket"})
            )

        # 4. Conversion by social_media_engagement
        st.markdown("### Conversion by social media engagement")
        bins_social = [0, 10, 30, 60, 100, np.inf]
        labels_social = ["0–10", "11–30", "31–60", "61–100", "100+"]

        conv_vs_social = conversion_by_bin(
            df_model, "social_media_engagement", bins_social, labels_social
        )
        st.bar_chart(conv_vs_social.set_index("bin")["conversion_rate"])
        with st.expander("See underlying numbers for social media engagement"):
            st.dataframe(
                conv_vs_social.rename(columns={"bin": "social_media_engagement_bucket"})
            )

        # 5. Conversion by response_time_hours
        st.markdown("### Conversion by response time")
        bins_resp = [0, 1, 4, 12, 24, np.inf]
        labels_resp = ["≤1h", "1–4h", "4–12h", "12–24h", ">24h"]

        conv_vs_resp = conversion_by_bin(
            df_model, "response_time_hours", bins_resp, labels_resp
        )
        st.bar_chart(conv_vs_resp.set_index("bin")["conversion_rate"])
        with st.expander("See underlying numbers for response time"):
            st.dataframe(
                conv_vs_resp.rename(columns={"bin": "response_time_bucket"})
            )

        # 6. Conversion by lead_status_score
        st.markdown("### Conversion by lead status (Cold/Warm/Hot)")
        conv_vs_lead = (
            df_model.groupby("lead_status_score")
            .agg(
                users=("user_id", "count"),
                converters=("converted_to_paid", "sum"),
            )
            .reset_index()
        )
        conv_vs_lead["conversion_rate"] = (
            conv_vs_lead["converters"] / conv_vs_lead["users"]
        )
        st.bar_chart(
            conv_vs_lead.set_index("lead_status_score")["conversion_rate"]
        )
        with st.expander("See underlying numbers for lead status"):
            st.dataframe(conv_vs_lead)

    # Feature importance chart
    st.markdown("---")
    if fi_df is not None:
        st.markdown("### Feature importance (Random Forest)")
        st.bar_chart(fi_df.set_index("feature")["importance"])
    else:
        st.info(
            "Feature importance file not found. Place 'feature_importance_rf_final.csv' next to app.py."
        )

# -------------------
# TAB 3: Industry Benchmarks
# -------------------
with tab_benchmarks:
    st.subheader("Industry Benchmarks (trial-to-paid)")

    bench = pd.DataFrame(
        {
            "scenario": [
                "This synthetic dataset",
                "Opt-in free trial (B2B SaaS, typical)",
                "Opt-out free trial (card required, B2B SaaS)",
            ],
            "conversion_rate": [
                dataset_conv_rate if dataset_conv_rate is not None else np.nan,
                0.18,  # ~18% for opt-in trials
                0.49,  # ~49% for opt-out trials
            ],
            "source": [
                "Kaggle synthetic",
                "Public SaaS benchmarks (opt-in)",
                "Public SaaS benchmarks (opt-out)",
            ],
        }
    )

    bench["conversion_rate (%)"] = (bench["conversion_rate"] * 100).round(1)

    st.dataframe(bench[["scenario", "conversion_rate (%)", "source"]])

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