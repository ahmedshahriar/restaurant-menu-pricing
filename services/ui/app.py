import json
import os
import time

import requests
import streamlit as st

st.set_page_config(page_title="Menu Price Prediction", page_icon="🍔", layout="centered")
st.title("🍔 Menu Price Prediction")
st.caption("Streamlit → APIM → Azure ML")

# ---- Constants / choices -----------------------------------------------------
CATEGORIES = ["Sandwiches", "Salads", "Wraps"]

STATE_LABELS = {
    "tx": "Texas",
    "va": "Virginia",
    "wa": "Washington",
    "wi": "Wisconsin",
    "ut": "Utah",
}


# Streamlit selectbox will display full names via format_func,
# but the returned value is the lowercase state code we need in payload.
STATE_OPTIONS = list(STATE_LABELS.keys())


def state_format(code: str) -> str:
    return f"{STATE_LABELS[code]} ({code.upper()})"


DEFAULT_INGREDIENTS = ["brisket", "bun", "bbq sauce"]

# ---- Config / secrets --------------------------------------------------------
APIM_URL = os.getenv("APIM_HOST", "")
APIM_SUBSCRIPTION_KEY = os.getenv("APIM_KEY", "")
AML_DEPLOYMENT = os.getenv("AML_DEPLOYMENT", "").strip()

with st.expander("Connection"):
    st.write("**APIM URL:**", APIM_URL or "❌ not set")
    st.write("**Subscription Key:**", "✅ loaded" if APIM_SUBSCRIPTION_KEY else "❌ missing")
    st.write("**Pinned Deployment (azureml-model-deployment):**", AML_DEPLOYMENT if AML_DEPLOYMENT else "Not pinned")

# ---- Form --------------------------------------------------------------------
with st.form("score_form"):
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Category", CATEGORIES, index=0)
        state_id = st.selectbox("State", STATE_OPTIONS, index=STATE_OPTIONS.index("tx"), format_func=state_format)
        city = st.text_input("City", "houston")

    with col2:
        price_range = st.selectbox("Price range", ["cheap", "moderate", "expensive"], index=0)
        density = st.number_input("Population density", value=1399.0, step=1.0, min_value=0.0)
        coli = st.number_input("Cost of living index", value=56.64, step=0.01, min_value=0.0)

    ingredients_raw = st.text_input("Ingredients (comma-separated)", ", ".join(DEFAULT_INGREDIENTS))

    submitted = st.form_submit_button("Predict")

if submitted:
    ingredients: list[str] = [x.strip() for x in ingredients_raw.split(",") if x.strip()]

    payload = {
        "input_data": {
            "columns": [
                "price_range",
                "state_id",
                "city",
                "density",
                "category",
                "ingredients",
                "cost_of_living_index",
            ],
            # state_id is the LOWERCASE code selected above (e.g., "tx")
            "data": [[price_range, state_id, city, float(density), category, ingredients, float(coli)]],
        }
    }

    if not APIM_URL or not APIM_SUBSCRIPTION_KEY:
        st.error("APIM_URL or APIM_SUBSCRIPTION_KEY is not configured.")
    else:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": APIM_SUBSCRIPTION_KEY,
        }
        # *** Make use of AML_DEPLOYMENT ***
        # If set (e.g., "green" or "blue"), APIM forwards it to AML to pin the deployment.
        if AML_DEPLOYMENT:
            headers["azureml-model-deployment"] = AML_DEPLOYMENT

        with st.spinner("Scoring…"):
            t0 = time.time()
            try:
                resp = requests.post(APIM_URL, headers=headers, data=json.dumps(payload), timeout=30)
                elapsed = time.time() - t0
                st.write(f"⏱️ {elapsed:.2f}s")

                ct = resp.headers.get("content-type", "")
                if ct.startswith("application/json"):
                    st.success("Prediction:")
                    st.json(resp.json())
                else:
                    st.code(resp.text, language="json")

                st.download_button(
                    "Download response JSON",
                    data=resp.text,
                    file_name="prediction.json",
                    mime="application/json",
                )
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.code(json.dumps(payload, indent=2), language="json")
