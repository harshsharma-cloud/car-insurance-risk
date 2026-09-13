"""
streamlit_app.py -- User-facing prediction interface 

This app allows a customers to enter their/vehicle profile information
and receive a predicted claim probability + risk level verdict.

Rules:
  - outcome is NEVER shown as an input field
  - The production pipeline is loaded once at startup (cached)
  - No model retraining occurs at any point
  - Invalid inputs are caught before prediction
  - Inputs are validated against known feature ranges

Run with:  streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src.predict import predict_single, get_pipeline, VALID_CATEGORIES, VALID_RANGES
from src.risk import assign_risk_level, build_verdict, get_threshold_config
from src.interpret import feature_importance_df

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Insurance Claim Risk Predictor",
    page_icon="car",
    layout="centered",
)

# ── Cache the pipeline at startup ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return get_pipeline()

pipeline = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Car Insurance Claim Risk Predictor")
st.markdown(
    "Enter a customer profile below to receive a predicted claim probability "
    "and risk level assessment. This tool uses a validated Logistic Regression "
    "model trained on historical claim data."
)

st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")

    age_labels = {
        0: "16-25",
        1: "26-39",
        2: "40-64",
        3: "65+",
    }
    age_display = st.selectbox(
        "Age Group",
        options=list(age_labels.values()),
        index=1,
        help="Select the customer's age bracket."
    )
    age = [k for k, v in age_labels.items() if v == age_display][0]

    gender_display = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        index=0,
        help="0 = Female, 1 = Male in the original data."
    )
    gender = 0 if gender_display == "Female" else 1

    married_display = st.selectbox(
        "Married",
        options=["No", "Yes"],
        index=0,
    )
    married = 0 if married_display == "No" else 1

    children_display = st.selectbox(
        "Has Children",
        options=["No", "Yes"],
        index=0,
    )
    children = 0 if children_display == "No" else 1

    education = st.selectbox(
        "Education Level",
        options=VALID_CATEGORIES["education"],
        index=0,
    )

    income = st.selectbox(
        "Income Class",
        options=VALID_CATEGORIES["income"],
        index=0,
    )

with col2:
    st.markdown("**Driving & Vehicle**")

    driving_experience = st.selectbox(
        "Driving Experience",
        options=VALID_CATEGORIES["driving_experience"],
        index=0,
        help="Years of driving experience."
    )

    vehicle_ownership_display = st.selectbox(
        "Vehicle Ownership",
        options=["Does not own vehicle", "Owns vehicle"],
        index=1,
    )
    vehicle_ownership = 0 if vehicle_ownership_display == "Does not own vehicle" else 1

    vehicle_year = st.selectbox(
        "Vehicle Year",
        options=VALID_CATEGORIES["vehicle_year"],
        index=1,
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        options=VALID_CATEGORIES["vehicle_type"],
        index=0,
    )

    credit_score = st.slider(
        "Credit Score (normalised)",
        min_value=0.0,
        max_value=1.0,
        value=0.52,
        step=0.01,
        help="Normalised credit score between 0 and 1. Median is ~0.52."
    )

    annual_mileage = st.slider(
        "Annual Mileage",
        min_value=2000,
        max_value=22000,
        value=12000,
        step=500,
        help="Estimated annual mileage. Dataset range: 2,000 - 22,000."
    )

st.markdown("**Driving Record**")
rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    speeding_violations = st.number_input(
        "Speeding Violations",
        min_value=0,
        max_value=25,
        value=0,
        step=1,
    )

with rec_col2:
    duis = st.number_input(
        "DUIs",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

with rec_col3:
    past_accidents = st.number_input(
        "Past Accidents",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict Claim Risk", type="primary", use_container_width=True):

    # Build the profile dict
    profile = {
        "age": age,
        "gender": gender,
        "driving_experience": driving_experience,
        "education": education,
        "income": income,
        "credit_score": credit_score,
        "vehicle_ownership": vehicle_ownership,
        "vehicle_year": vehicle_year,
        "married": married,
        "children": children,
        "annual_mileage": annual_mileage,
        "vehicle_type": vehicle_type,
        "speeding_violations": speeding_violations,
        "duis": duis,
        "past_accidents": past_accidents,
    }

    try:
        result = predict_single(profile)
        verdict = build_verdict(result["claim_probability"])

        prob = verdict["claim_probability"]
        pct = verdict["claim_probability_pct"]
        risk = verdict["risk_level"]

        # ── Results display ──────────────────────────────────────────────
        st.subheader("Prediction Result")

        # Risk level colour mapping
        risk_colors = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red",
        }
        risk_emojis = {
            "LOW": "white_check_mark",
            "MEDIUM": "warning",
            "HIGH": "rotating_light",
        }

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Claim Probability", pct)
        m2.metric("Risk Level", risk)
        m3.metric("Model Confidence", f"{max(prob, 1-prob)*100:.0f}%")

        # Verdict box
        if risk == "LOW":
            st.success(f":{risk_emojis[risk]}: **{risk} RISK** -- {verdict['verdict_text']}")
        elif risk == "MEDIUM":
            st.warning(f":{risk_emojis[risk]}: **{risk} RISK** -- {verdict['verdict_text']}")
        else:
            st.error(f":{risk_emojis[risk]}: **{risk} RISK** -- {verdict['verdict_text']}")

        # ── Key predictive factors ───────────────────────────────────────
        with st.expander("Key Predictive Factors", expanded=True):
            st.markdown(
                "The table below shows the most important features in the model. "
                "These indicate **statistical associations**, not causal relationships."
            )

            imp_df = feature_importance_df(pipeline)
            top10 = imp_df.head(10).copy()
            top10["direction"] = top10["coefficient"].apply(
                lambda c: "Higher risk (+)" if c > 0 else "Lower risk (-)"
            )
            # Build a fully string-typed DataFrame — avoids ALL Arrow issues
            display_df = pd.DataFrame({
                "Rank":      top10["rank"].astype(int).astype(str).tolist(),
                "Feature":   top10["feature"].tolist(),
                "Coeff":     top10["coefficient"].apply(lambda x: f"{x:+.4f}").tolist(),
                "Abs Coeff": top10["importance"].apply(lambda x: f"{x:.4f}").tolist(),
                "Direction": top10["direction"].tolist(),
            })
            st.table(display_df)

            st.caption(
                "These coefficients are from a logistic regression model trained on "
                "standardised features. Larger absolute values indicate stronger "
                "predictive association. This does NOT imply causation."
            )

        # ── Profile summary ──────────────────────────────────────────────
        with st.expander("Input Profile Summary"):
            profile_display = {
                "Age Group": age_display,
                "Gender": gender_display,
                "Married": married_display,
                "Has Children": children_display,
                "Education": education,
                "Income": income,
                "Driving Experience": driving_experience,
                "Vehicle Ownership": vehicle_ownership_display,
                "Vehicle Year": vehicle_year,
                "Vehicle Type": vehicle_type,
                "Credit Score": f"{credit_score:.2f}",
                "Annual Mileage": f"{annual_mileage:,}",
                "Speeding Violations": str(int(speeding_violations)),
                "DUIs": str(int(duis)),
                "Past Accidents": str(int(past_accidents)),
            }
            # All values forced to str — avoids Arrow mixed-type error
            st.table(pd.DataFrame(
                [(str(k), str(v)) for k, v in profile_display.items()],
                columns=["Feature", "Value"]
            ))

    except ValueError as e:
        st.error(f"Input validation error: {e}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()

thresholds = get_threshold_config()
st.caption(
    f"**Risk Thresholds** (evidence-based, validated on test set): "
    f"LOW < {thresholds['low_threshold']:.0%} | "
    f"MEDIUM [{thresholds['low_threshold']:.0%}, {thresholds['high_threshold']:.0%}) | "
    f"HIGH >= {thresholds['high_threshold']:.0%}"
)
st.caption(
    "This tool provides statistical risk predictions based on historical data. "
    "It does NOT establish causation and should not be used as the sole basis "
    "for underwriting decisions."
)
