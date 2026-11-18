"""
Consumer Complaint Classification System - Main Streamlit Application
======================================================================
A professional multi-page Streamlit dashboard for classifying and
analyzing consumer financial complaints using NLP and ML.
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Consumer Complaint Classifier",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
# Injected as a continuous, compact stylesheet without blank lines to prevent
# Streamlit markdown from breaking raw HTML blocks and rendering text.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #ffffff !important; font-size: 15px; font-weight: 500; }
.kpi-card { background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%); border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); padding: 20px 24px; text-align: center; color: #ffffff; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08); }
.kpi-card h2 { font-size: 2.2rem; font-weight: 700; color: #e94560; margin: 0; }
.kpi-card p { font-size: 0.85rem; color: #a0aec0; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.08em; }
.prediction-card { background: linear-gradient(135deg, #0f3460 0%, #533483 100%); border-radius: 16px; padding: 28px 32px; color: #ffffff; box-shadow: 0 10px 40px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); margin-top: 16px; }
.prediction-card h2 { font-size: 1.6rem; font-weight: 700; color: #53d8fb; margin-bottom: 4px; }
.prediction-card .label { font-size: 0.8rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.08em; }
.confidence-badge { display: inline-block; background: rgba(83, 216, 251, 0.2); border: 1px solid #53d8fb; color: #53d8fb; border-radius: 20px; padding: 4px 16px; font-weight: 600; font-size: 1rem; margin-top: 8px; }
.section-header { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; padding-bottom: 8px; border-bottom: 3px solid #e94560; margin-bottom: 20px; }
.best-badge { background: #38a169; color: white; border-radius: 12px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }
.info-box { background: #ebf8ff; border-left: 4px solid #3182ce; border-radius: 4px; padding: 12px 16px; margin: 10px 0; color: #2b6cb0; font-size: 0.9rem; }
"""
st.markdown(f"<style>{CUSTOM_CSS.strip()}</style>", unsafe_allow_html=True)

# ── Path constants ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "complaints.csv")

# ── Analytics imports ──────────────────────────────────────────────────────
try:
    from analytics import (
        get_category_distribution,
        get_monthly_trend,
        get_company_analysis,
        get_status_distribution,
        get_complaint_heatmap,
        get_state_distribution,
        get_yearly_trend,
    )
    _analytics_ok = True
except ImportError:
    _analytics_ok = False

# ── Preprocessing import ───────────────────────────────────────────────────
try:
    from preprocessing import preprocess_pipeline
    _preprocessing_ok = True
except ImportError:
    _preprocessing_ok = False
    def preprocess_pipeline(text):
        return text.lower().strip()

# ── Cached resource loaders ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the trained model, vectorizer, and metadata from disk."""
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        metadata = joblib.load(METADATA_PATH)
        return model, vectorizer, metadata
    except FileNotFoundError:
        return None, None, None
    except Exception as exc:
        st.warning(f"Could not load model: {exc}")
        return None, None, None


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """Load the complaints CSV from disk."""
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        return df
    except FileNotFoundError:
        return None
    except Exception as exc:
        st.warning(f"Could not load dataset: {exc}")
        return None


# ── Load everything ────────────────────────────────────────────────────────
model, vectorizer, metadata = load_model()
df = load_data()

model_loaded = model is not None and vectorizer is not None
data_loaded = df is not None

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#e94560;text-align:center;font-size:1.3rem;'>🛡️ Complaint Classifier</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home Dashboard",
            "🔍 Complaint Classification",
            "📊 Complaint Analytics",
            "📋 Dataset Explorer",
            "📈 Model Performance",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    # Status indicators
    st.markdown("**System Status**")
    if model_loaded:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not loaded")
    if data_loaded:
        st.success(f"✅ Data: {len(df):,} rows")
    else:
        st.warning("⚠️ No dataset found")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home Dashboard":

    st.markdown(
        "<h1 style='font-size:2.2rem;font-weight:700;color:#1a1a2e;'>🛡️ Consumer Complaint Classification System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:1.1rem;color:#718096;margin-top:-12px;'>AI-Powered Complaint Analysis & Categorization</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── KPI section ────────────────────────────────────────────────────────
    total_complaints = len(df) if data_loaded else 0
    total_categories = df["Product"].nunique() if data_loaded and "Product" in df.columns else 0
    top_category = (
        df["Product"].value_counts().idxmax()
        if data_loaded and "Product" in df.columns
        else "N/A"
    )
    model_accuracy = "N/A"
    model_name_display = "N/A"
    if metadata:
        try:
            best = metadata.get("model_name", "N/A")
            model_name_display = best
            if best == "Logistic Regression":
                model_accuracy = f"{metadata['lr_metrics']['accuracy']:.2%}"
            else:
                model_accuracy = f"{metadata['nb_metrics']['accuracy']:.2%}"
        except Exception:
            pass

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📋 Total Complaints", f"{total_complaints:,}")
    with k2:
        st.metric("🗂️ Total Categories", f"{total_categories}")
    with k3:
        st.metric("🏆 Top Category", top_category[:25] + "…" if len(str(top_category)) > 25 else top_category)
    with k4:
        st.metric("🎯 Model Accuracy", model_accuracy)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────
    if data_loaded and _analytics_ok:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Complaints by Category")
            fig_cat = get_category_distribution(df)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Category data unavailable.")

        with c2:
            st.markdown("#### 📈 Monthly Complaint Trend")
            fig_trend = get_monthly_trend(df)
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Date data unavailable.")
    else:
        st.info("Load a dataset to see charts. Place `complaints.csv` in the `data/` folder.")

    # ── Key Insights ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown(
            f"""<div class='info-box'>
            <b>🏷️ Top Category</b><br/>
            The most common complaint category is <b>{top_category}</b>.
            It represents the highest volume of consumer grievances in this dataset.
            </div>""",
            unsafe_allow_html=True,
        )
    with col_i2:
        st.markdown(
            f"""<div class='info-box'>
            <b>📋 Dataset Size</b><br/>
            The dataset contains <b>{total_complaints:,}</b> consumer complaints
            spanning <b>{total_categories}</b> distinct product categories.
            </div>""",
            unsafe_allow_html=True,
        )
    with col_i3:
        st.markdown(
            f"""<div class='info-box'>
            <b>🤖 ML Model</b><br/>
            The active classifier is <b>{model_name_display}</b>.
            It was trained on TF-IDF features with n-gram range (1,2).
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 – COMPLAINT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Complaint Classification":

    st.markdown(
        "<h1 class='section-header'>🔍 Classify a Consumer Complaint</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Enter or select a consumer complaint below and our AI model will automatically
        categorize it into one of the trained product categories. The classifier uses
        **TF-IDF vectorization** combined with a **machine-learning model** trained on
        thousands of real CFPB complaints.
        """
    )
    st.markdown("---")

    SAMPLE_COMPLAINTS = {
        "💳 Overdraft Fees (Banking)": (
            "I was charged multiple overdraft fees for the same transaction. "
            "The bank refused to refund these charges even after I explained the situation."
        ),
        "📈 Interest Rate Hike (Credit Card)": (
            "My credit card company increased my interest rate without proper notification. "
            "I have always paid on time."
        ),
        "📞 Aggressive Debt Collector": (
            "A debt collector keeps calling me at work after I told them not to. "
            "They are very aggressive and threatening."
        ),
        "🏠 Mortgage Payment Error": (
            "My mortgage servicer applied my payment to the wrong account and reported me as "
            "late to the credit bureaus."
        ),
        "📄 Credit Report Dispute": (
            "I found incorrect information on my credit report. "
            "The company refuses to investigate my dispute."
        ),
        "🎓 Student Loan Plan Change": (
            "The student loan servicer changed my repayment plan without my consent, "
            "increasing my monthly payment significantly."
        ),
        "✏️ Enter your own complaint": None,
    }

    sample_choice = st.selectbox(
        "Select a sample complaint or enter your own:",
        list(SAMPLE_COMPLAINTS.keys()),
    )

    if SAMPLE_COMPLAINTS[sample_choice] is None:
        complaint_text = st.text_area(
            "Your complaint:",
            height=160,
            placeholder="Describe your consumer complaint here…",
        )
    else:
        complaint_text = st.text_area(
            "Complaint text (editable):",
            value=SAMPLE_COMPLAINTS[sample_choice],
            height=160,
        )

    classify_btn = st.button("🔮 Classify Complaint", type="primary", use_container_width=False)

    if classify_btn:
        if not model_loaded:
            st.error("❌ Model not loaded. Please run `python train_model.py` first to train the model.")
        elif not complaint_text.strip():
            st.warning("⚠️ Please enter a complaint before classifying.")
        else:
            with st.spinner("Analyzing complaint…"):
                try:
                    processed = preprocess_pipeline(complaint_text)
                    vec = vectorizer.transform([processed])
                    prediction = model.predict(vec)[0]

                    # Try predict_proba
                    has_proba = hasattr(model, "predict_proba")
                    if has_proba:
                        proba = model.predict_proba(vec)[0]
                        classes = model.classes_
                        # Decode if LabelEncoder was used
                        # train_model.py saves the key as "label_encoder_classes"
                        _lec_key = "label_encoder_classes" if metadata and "label_encoder_classes" in metadata else "label_classes"
                        if metadata and _lec_key in metadata:
                            label_classes = metadata[_lec_key]
                            class_names = list(label_classes)
                            prediction_label = label_classes[prediction] if isinstance(prediction, (int, np.integer)) else prediction
                        else:
                            class_names = list(classes)
                            prediction_label = prediction
                        confidence = float(np.max(proba)) * 100

                        # Top 3
                        sorted_idx = np.argsort(proba)[::-1][:3]
                        top3 = [(class_names[i] if i < len(class_names) else str(classes[i]), float(proba[i]) * 100) for i in sorted_idx]
                    else:
                        has_proba = False
                        _lec_key = "label_encoder_classes" if metadata and "label_encoder_classes" in metadata else "label_classes"
                        if metadata and _lec_key in metadata:
                            label_classes = metadata[_lec_key]
                            prediction_label = label_classes[prediction] if isinstance(prediction, (int, np.integer)) else prediction
                        else:
                            prediction_label = str(prediction)
                        confidence = None
                        top3 = []

                    # Display result card
                    conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"
                    st.markdown(
                        f"""
                        <div class='prediction-card'>
                            <div class='label'>Predicted Category</div>
                            <h2>{prediction_label}</h2>
                            <div class='label'>Confidence Score</div>
                            <span class='confidence-badge'>🎯 {conf_str}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if top3:
                        st.markdown("#### 📊 Top 3 Predicted Categories")
                        for name, pct in top3:
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.progress(pct / 100, text=name)
                            with col_b:
                                st.markdown(f"**{pct:.1f}%**")

                except Exception as exc:
                    st.error(f"Classification error: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 – COMPLAINT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Complaint Analytics":

    st.markdown(
        "<h1 class='section-header'>📊 Complaint Analytics Dashboard</h1>",
        unsafe_allow_html=True,
    )

    if not data_loaded:
        st.warning("⚠️ No dataset found. Please add `complaints.csv` to the `data/` folder.")
        st.stop()

    if not _analytics_ok:
        st.error("❌ analytics.py could not be imported. Ensure it exists in the project directory.")
        st.stop()

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔽 Filters", expanded=True):
        f1, f2 = st.columns(2)

        filtered_df = df.copy()

        with f1:
            if "Date received" in df.columns:
                try:
                    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
                    min_date = df["Date received"].min()
                    max_date = df["Date received"].max()
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                    )
                    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                        start, end = date_range
                        filtered_df = filtered_df[
                            (filtered_df["Date received"] >= pd.Timestamp(start))
                            & (filtered_df["Date received"] <= pd.Timestamp(end))
                        ]
                except Exception:
                    pass

        with f2:
            if "Product" in df.columns:
                cats = sorted(df["Product"].dropna().unique().tolist())
                selected_cats = st.multiselect("Product Categories", cats, default=[])
                if selected_cats:
                    filtered_df = filtered_df[filtered_df["Product"].isin(selected_cats)]

    st.markdown(f"**Showing {len(filtered_df):,} complaints** after filters.")
    st.markdown("---")

    # ── 2×2 chart grid ─────────────────────────────────────────────────────
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown("#### 🗂️ Category Distribution")
        try:
            fig = get_category_distribution(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available.")
        except Exception as e:
            st.warning(f"Chart error: {e}")

    with row1_c2:
        st.markdown("#### 📈 Monthly Trend")
        try:
            fig = get_monthly_trend(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No date data available.")
        except Exception as e:
            st.warning(f"Chart error: {e}")

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown("#### 🏢 Top Companies by Complaints")
        try:
            fig = get_company_analysis(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No company data available.")
        except Exception as e:
            st.warning(f"Chart error: {e}")

    with row2_c2:
        st.markdown("#### ⚖️ Company Response Status")
        try:
            fig = get_status_distribution(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No status data available.")
        except Exception as e:
            st.warning(f"Chart error: {e}")

    st.markdown("---")

    # ── Full-width charts ──────────────────────────────────────────────────
    st.markdown("#### 🗺️ Complaints by US State")
    try:
        fig = get_state_distribution(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No state data available.")
    except Exception as e:
        st.warning(f"Chart error: {e}")

    st.markdown("#### 🔥 Product × Company Heatmap")
    try:
        fig = get_complaint_heatmap(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for heatmap.")
    except Exception as e:
        st.warning(f"Chart error: {e}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 – DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":

    st.markdown(
        "<h1 class='section-header'>📋 Dataset Explorer</h1>",
        unsafe_allow_html=True,
    )

    tab_current, tab_upload = st.tabs(["📂 Current Dataset", "⬆️ Upload New Dataset"])

    # ── Tab 1: Current Dataset ─────────────────────────────────────────────
    with tab_current:
        if not data_loaded:
            st.warning("⚠️ No dataset found. Place `complaints.csv` in the `data/` folder.")
        else:
            # Dataset info expander
            with st.expander("ℹ️ Dataset Info", expanded=False):
                i1, i2, i3 = st.columns(3)
                with i1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with i2:
                    st.metric("Columns", df.shape[1])
                with i3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

                st.markdown("**Column Types**")
                dtype_df = pd.DataFrame(
                    {"Column": df.dtypes.index, "Type": df.dtypes.values.astype(str)}
                )
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)

            # Search & filter
            search_col, col_sel = st.columns([2, 2])
            with search_col:
                search_term = st.text_input("🔍 Search rows", placeholder="Type to filter any column…")
            with col_sel:
                selected_cols = st.multiselect(
                    "Columns to display",
                    options=df.columns.tolist(),
                    default=df.columns.tolist()[:6],
                )

            display_df = df.copy()

            if search_term.strip():
                str_cols = display_df.select_dtypes(include="object").columns
                mask = display_df[str_cols].apply(
                    lambda col: col.astype(str).str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                display_df = display_df[mask]

            if selected_cols:
                display_df = display_df[selected_cols]

            st.markdown(f"**{len(display_df):,} rows displayed**")
            st.dataframe(display_df, use_container_width=True, height=380)

            # Missing value analysis
            st.markdown("#### 🕳️ Missing Value Analysis")
            missing_pct = (df.isnull().sum() / len(df) * 100).reset_index()
            missing_pct.columns = ["Column", "Missing %"]
            missing_pct = missing_pct[missing_pct["Missing %"] > 0].sort_values("Missing %", ascending=False)

            if not missing_pct.empty:
                fig_missing = px.bar(
                    missing_pct,
                    x="Column",
                    y="Missing %",
                    title="Missing Data Percentage per Column",
                    color="Missing %",
                    color_continuous_scale="Reds",
                    template="plotly_white",
                )
                fig_missing.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

            # Download
            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Filtered Data as CSV",
                data=csv_bytes,
                file_name="filtered_complaints.csv",
                mime="text/csv",
            )

    # ── Tab 2: Upload New Dataset ──────────────────────────────────────────
    with tab_upload:
        st.markdown(
            """
            #### How to get the CFPB Dataset
            1. Visit the [CFPB Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
            2. Click **Download** and select **CSV** format
            3. Upload the downloaded file below

            > **Required columns:** `Consumer complaint narrative`, `Product`
            """
        )

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file, low_memory=False)
                st.success(f"✅ Uploaded: **{uploaded_file.name}** — {len(upload_df):,} rows, {upload_df.shape[1]} columns")

                # Preview
                st.markdown("**Preview (first 5 rows)**")
                st.dataframe(upload_df.head(), use_container_width=True)

                # Basic info
                u1, u2, u3 = st.columns(3)
                with u1:
                    st.metric("Rows", f"{upload_df.shape[0]:,}")
                with u2:
                    st.metric("Columns", upload_df.shape[1])
                with u3:
                    st.metric("Missing Cells", f"{upload_df.isnull().sum().sum():,}")

                # Column selector for classification
                st.markdown("---")
                st.markdown("#### 🤖 Run Classification on Uploaded Data")

                str_cols = upload_df.select_dtypes(include="object").columns.tolist()

                cc1, cc2 = st.columns(2)
                with cc1:
                    text_col = st.selectbox("Text column (complaint narrative)", str_cols)
                with cc2:
                    cat_col = st.selectbox("Category column (optional, for comparison)", ["None"] + str_cols)

                run_classify = st.button("🔮 Classify All Rows", type="primary")

                if run_classify:
                    if not model_loaded:
                        st.error("❌ Model not loaded. Train the model first.")
                    elif text_col not in upload_df.columns:
                        st.warning("Selected text column not found.")
                    else:
                        with st.spinner("Classifying…"):
                            try:
                                texts = upload_df[text_col].fillna("").astype(str).tolist()
                                processed_texts = [preprocess_pipeline(t) for t in texts]
                                vecs = vectorizer.transform(processed_texts)
                                preds = model.predict(vecs)

                                if metadata and "label_classes" in metadata:
                                    lc = metadata["label_classes"]
                                    pred_labels = [lc[p] if isinstance(p, (int, np.integer)) else p for p in preds]
                                else:
                                    pred_labels = list(preds)

                                result_df = upload_df.copy()
                                result_df["Predicted_Category"] = pred_labels

                                st.success(f"✅ Classified {len(result_df):,} complaints!")
                                st.dataframe(result_df[([text_col] + (["Predicted_Category"]))].head(20), use_container_width=True)

                                # Download results
                                res_csv = result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "⬇️ Download Classified Results",
                                    data=res_csv,
                                    file_name="classified_complaints.csv",
                                    mime="text/csv",
                                )
                            except Exception as exc:
                                st.error(f"Classification failed: {exc}")

            except Exception as exc:
                st.error(f"Could not read file: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown(
        "<h1 class='section-header'>📈 Model Performance</h1>",
        unsafe_allow_html=True,
    )

    if not metadata:
        st.warning(
            "⚠️ Model metadata not found. Please run `python train_model.py` to train the models "
            "and generate performance metrics."
        )
        st.stop()

    # Extract metrics
    lr_metrics = metadata.get("lr_metrics", {})
    nb_metrics = metadata.get("nb_metrics", {})
    best_model_name = metadata.get("model_name", "N/A")

    lr_acc = lr_metrics.get("accuracy", 0)
    lr_pre = lr_metrics.get("precision", 0)
    lr_rec = lr_metrics.get("recall", 0)
    lr_f1  = lr_metrics.get("f1", 0)

    nb_acc = nb_metrics.get("accuracy", 0)
    nb_pre = nb_metrics.get("precision", 0)
    nb_rec = nb_metrics.get("recall", 0)
    nb_f1  = nb_metrics.get("f1", 0)

    lr_is_best = best_model_name == "Logistic Regression"

    # ── Metrics side-by-side ───────────────────────────────────────────────
    col_lr, col_nb = st.columns(2)

    with col_lr:
        badge = "<span class='best-badge'>⭐ Best Model</span>" if lr_is_best else ""
        st.markdown(f"### 📘 Logistic Regression {badge}", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        with m1:
            st.metric("Accuracy", f"{lr_acc:.4f}", delta=f"{lr_acc - nb_acc:+.4f} vs NB")
        with m2:
            st.metric("Precision", f"{lr_pre:.4f}", delta=f"{lr_pre - nb_pre:+.4f} vs NB")
        with m3:
            st.metric("Recall", f"{lr_rec:.4f}", delta=f"{lr_rec - nb_rec:+.4f} vs NB")
        with m4:
            st.metric("F1-Score", f"{lr_f1:.4f}", delta=f"{lr_f1 - nb_f1:+.4f} vs NB")

    with col_nb:
        badge = "<span class='best-badge'>⭐ Best Model</span>" if not lr_is_best else ""
        st.markdown(f"### 📗 Naive Bayes {badge}", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        with m1:
            st.metric("Accuracy", f"{nb_acc:.4f}", delta=f"{nb_acc - lr_acc:+.4f} vs LR")
        with m2:
            st.metric("Precision", f"{nb_pre:.4f}", delta=f"{nb_pre - lr_pre:+.4f} vs LR")
        with m3:
            st.metric("Recall", f"{nb_rec:.4f}", delta=f"{nb_rec - lr_rec:+.4f} vs LR")
        with m4:
            st.metric("F1-Score", f"{nb_f1:.4f}", delta=f"{nb_f1 - lr_f1:+.4f} vs LR")

    st.markdown("---")

    # ── Model comparison bar chart ─────────────────────────────────────────
    st.markdown("#### 📊 Model Comparison")
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_vals = [lr_acc, lr_pre, lr_rec, lr_f1]
    nb_vals = [nb_acc, nb_pre, nb_rec, nb_f1]

    fig_comp = go.Figure(data=[
        go.Bar(name="Logistic Regression", x=metrics_names, y=lr_vals,
               marker_color="#3182ce", text=[f"{v:.3f}" for v in lr_vals], textposition="outside"),
        go.Bar(name="Naive Bayes", x=metrics_names, y=nb_vals,
               marker_color="#38a169", text=[f"{v:.3f}" for v in nb_vals], textposition="outside"),
    ])
    fig_comp.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.15], title="Score"),
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
        title="Logistic Regression vs. Naive Bayes — All Metrics",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Classification report ──────────────────────────────────────────────
    best_metrics = lr_metrics if lr_is_best else nb_metrics
    with st.expander("📋 Classification Report (Best Model)", expanded=True):
        cr = best_metrics.get("classification_report") or metadata.get("lr_classification_report" if lr_is_best else "nb_classification_report")
        if cr and isinstance(cr, dict):
            try:
                cr_df = pd.DataFrame(cr).T
                # Format float scores nicely
                format_fn = lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else (f"{int(x)}" if isinstance(x, (int, np.integer)) else str(x))
                if hasattr(cr_df, "map"):
                    cr_df = cr_df.map(format_fn)
                else:
                    cr_df = cr_df.applymap(format_fn)
                st.dataframe(cr_df, use_container_width=True)
            except Exception as e:
                st.json(cr)
        else:
            st.info("Classification report not available in metadata.")

    # ── Confusion matrix ───────────────────────────────────────────────────
    st.markdown("#### 🔲 Confusion Matrix (Best Model)")
    cm = best_metrics.get("confusion_matrix") or metadata.get("lr_confusion_matrix" if lr_is_best else "nb_confusion_matrix")
    categories = metadata.get("categories") or metadata.get("label_encoder_classes")

    if cm is not None:
        try:
            cm_arr = np.array(cm)
            labels = categories if categories else [str(i) for i in range(cm_arr.shape[0])]
            fig_cm = px.imshow(
                cm_arr,
                x=labels,
                y=labels,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
                title=f"Confusion Matrix — {best_model_name}",
                aspect="auto",
                text_auto=True,
            )
            fig_cm.update_xaxes(tickangle=-40)
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render confusion matrix: {exc}")
    else:
        st.info("Confusion matrix data not found in metadata. Train the model to generate it.")

    # ── About the models ───────────────────────────────────────────────────
    with st.expander("📖 About the Models", expanded=False):
        ab1, ab2 = st.columns(2)
        with ab1:
            st.markdown("""
**📘 Logistic Regression**

Logistic Regression is a linear classification algorithm that models the probability
of a sample belonging to a particular class. In text classification:

- Works well with high-dimensional, sparse TF-IDF features
- Outputs calibrated probabilities via softmax (multi-class)
- Regularization (C parameter) prevents overfitting
- Fast training and highly interpretable weights
- Often the strongest baseline for NLP classification tasks
""")
        with ab2:
            st.markdown("""
**📗 Multinomial Naive Bayes**

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the
"naïve" conditional independence assumption between features:

- Extremely fast — works well even with limited data
- Naturally suited to word-count / TF-IDF feature matrices
- The `alpha` (Laplace smoothing) parameter avoids zero probabilities
- Performs well on short, noisy text
- Less accurate than LR on large datasets but very memory-efficient
""")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 – ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":

    st.markdown(
        "<h1 class='section-header'>ℹ️ About This Project</h1>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
### 🎯 Project Overview

**Consumer Complaint Classification System**

This project was developed as part of a **Web Data Mining Lab** assignment.
It demonstrates the full pipeline of an NLP + machine-learning application
applied to real-world consumer financial data from the CFPB.

**Purpose:**
- Automatically categorize consumer financial complaints
- Identify patterns in complaint data across categories, companies, and time
- Compare Logistic Regression vs. Naive Bayes classifiers
- Provide an interactive analytics dashboard

**Academic Context:**
- 🏫 Web Data Mining Laboratory Project
- 🎓 Final-year undergraduate / postgraduate level
- 📅 Built with Python & open-source libraries
""")

    with col_b:
        st.markdown("""
### 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| 🐍 Python 3.10+ | Core language |
| 🌊 Streamlit | Web dashboard |
| 🤖 scikit-learn | ML models |
| 📊 Plotly | Interactive charts |
| 🐼 pandas | Data wrangling |
| 🔢 NumPy | Numerical ops |
| 📝 NLTK | NLP preprocessing |
| 💾 joblib | Model serialisation |

**ML Pipeline:**
- 🧹 Text cleaning & stopword removal
- 🔢 TF-IDF Vectorization (n-gram 1–2)
- 📦 Label Encoding
- 🤖 Logistic Regression (C=5)
- 🤖 Multinomial Naive Bayes (α=0.1)
- 📏 Weighted F1 model selection
""")

    with col_c:
        st.markdown("""
### 📋 Dataset Info

**Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

The Consumer Financial Protection Bureau (CFPB) publishes a public database
of consumer complaints about financial products and services.

**Key Columns Used:**
- `Consumer complaint narrative` — raw text
- `Product` — category label
- `Company` — financial institution
- `State` — US state code
- `Date received` — complaint date
- `Company response to consumer` — resolution status

**Scale:**
- 3+ million total records (full dataset)
- Top 8 product categories used for training
- Up to 2,000 samples per category (balanced)

**License:** Public domain (US Government data)
""")

    st.markdown("---")

    # ── Web Data Mining concepts ───────────────────────────────────────────
    st.markdown("### 🌐 Web Data Mining Concepts Demonstrated")
    wdm1, wdm2, wdm3 = st.columns(3)
    with wdm1:
        st.markdown("""
**Data Acquisition**
- Downloading large CSV from CFPB portal
- Handling missing values and noise
- Data type inference and parsing
""")
    with wdm2:
        st.markdown("""
**Text Mining & NLP**
- Tokenisation & lowercasing
- Stopword removal (NLTK)
- TF-IDF feature extraction
- N-gram language modelling
""")
    with wdm3:
        st.markdown("""
**Machine Learning**
- Supervised multi-class classification
- Train/test split & stratified sampling
- Accuracy, Precision, Recall, F1 evaluation
- Confusion matrix & classification report
""")

    st.markdown("---")

    # ── Deployment & GitHub ────────────────────────────────────────────────
    dep1, dep2 = st.columns(2)
    with dep1:
        st.markdown("""
### 🚀 Deployment Info

This app is built with **Streamlit** and can be deployed to:

- 🌐 [Streamlit Community Cloud](https://streamlit.io/cloud) — free hosting
- ☁️ Google Cloud Run / AWS / Azure — containerised deployment
- 🐳 Docker — `docker build` with a simple Dockerfile

**To run locally:**
```bash
pip install -r requirements.txt
python train_model.py   # train the model first
streamlit run app.py
```
""")
    with dep2:
        st.markdown("""
### 👤 Contact & Author

| Field | Detail |
|---|---|
| 📌 Project | Consumer Complaint Classifier |
| 🏫 Context | Web Data Mining Lab |
| 📦 Stack | Python · Streamlit · scikit-learn |
| 📂 Data | CFPB Public Database |
| 📄 License | MIT |

> **Note:** This system is for educational purposes.
> Model predictions are based on training data and may not be
> perfectly accurate on all complaint types.

---
*Built with ❤️ using Python and Streamlit*
""")
