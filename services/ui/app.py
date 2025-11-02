import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Menu Price Prediction", page_icon="🍔", layout="centered")
st.title("🍔 Menu Price Prediction")
st.caption("Streamlit frontend → Azure APIM gateway → Azure ML scoring endpoint")

# --- Constants / choices ---
CATEGORIES = ["Sandwiches", "Salads", "Wraps"]

STATE_NAME_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}

CITY_DATA_DIR = Path(os.getenv("DATA_DIR", "services/common/data"))
STATE_CITY_FILE = CITY_DATA_DIR / "state_city_map.json"


@st.cache_data
def load_state_city_map():
    try:
        with STATE_CITY_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        st.error(
            f"Required file '{STATE_CITY_FILE}' not found. Please run the export script to generate it before starting the app."
        )
        raw = {}
    except json.JSONDecodeError:
        st.error(f"File '{STATE_CITY_FILE}' is not a valid JSON file. Please check or regenerate the file.")
        raw = {}

    # Normalize to: {state: {city: {"density": float, "cost_of_living_index": float}}}
    norm = {}
    for state, cities in (raw or {}).items():
        s = str(state).strip().lower()
        norm[s] = {}
        for city, meta in (cities or {}).items():
            c = str(city).strip().lower()
            m = meta or {}
            norm[s][c] = {
                "density": float(m.get("density", 0) or 0),
                "cost_of_living_index": float(m.get("cost_of_living_index", 0) or 0),
            }
    return norm


# Load state-city mapping from JSON
STATE_CITY_MAP = load_state_city_map()

# Dynamic states from JSON (keys are state names in your file)
STATE_NAMES = sorted(STATE_CITY_MAP.keys())  # e.g. ["texas","utah","virginia",...]


# Format state name for display
def state_format(name: str) -> str:
    return name.title()


DEFAULT_INGREDIENTS = ["whole grain bread", "tomato", "carrots", "red onions", "bbq sauce", "lettuce", "cheddar cheese"]

# ---- Config / secrets --------------------------------------------------------
APIM_URL = os.getenv("APIM_HOST", "")
APIM_SUBSCRIPTION_KEY = os.getenv("APIM_KEY", "")
AML_DEPLOYMENT = os.getenv("AML_DEPLOYMENT", "").strip()

with st.expander("Connection"):
    st.write("**APIM URL:**", "✅ loaded" if APIM_URL else "❌ missing")
    st.write("**Subscription Key:**", "✅ loaded" if APIM_SUBSCRIPTION_KEY else "❌ missing")
    st.write("**Pinned Deployment (azureml-model-deployment):**", AML_DEPLOYMENT if AML_DEPLOYMENT else "Not pinned")

# --- Form ----

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Category", CATEGORIES, index=0)

        # Select state by NAME (from JSON), then city based on that state
        state_name = st.selectbox("State", STATE_NAMES, index=STATE_NAMES.index("texas"), format_func=state_format)

        # Cities are the keys of the nested dict for that state
        cities_for_state = sorted(STATE_CITY_MAP.get(state_name, {}).keys())
        if cities_for_state:
            default_city = "houston" if "houston" in cities_for_state else cities_for_state[0]
            city = st.selectbox("City", cities_for_state, index=cities_for_state.index(default_city))
        else:
            st.caption("No cities found for this state in your dataset. Type one:")
            city = st.text_input("City", "").strip().lower()

    with col2:
        # Look up defaults from the nested JSON
        meta = STATE_CITY_MAP.get(state_name, {}).get(city, {})
        density_default = float(meta.get("density", 1399.0) or 1399.0)
        coli_default = float(meta.get("cost_of_living_index", 56.64) or 56.64)

        price_range = st.selectbox("Price range", ["cheap", "moderate", "expensive"], index=0)
        density = st.number_input("Population density", value=density_default, step=1.0, min_value=0.0)
        coli = st.number_input("Cost of living index", value=coli_default, step=0.01, min_value=0.0)

    ingredients_raw = st.text_input("Ingredients (comma-separated)", ", ".join(DEFAULT_INGREDIENTS))

    # Button instead of form submit
    submitted = st.button("Predict")


# clear prediction if inputs changed
def _fp():
    return (
        category,
        state_name,
        city,
        price_range,
        float(density),
        float(coli),
        ingredients_raw,
    )


curr_fp = _fp()
if st.session_state.get("last_fp") != curr_fp:
    st.session_state["last_fp"] = curr_fp
    st.session_state.pop("prediction", None)

result_box = st.empty()

# --- Submit / scoring ---
if submitted:
    ingredients: list[str] = [x.strip() for x in ingredients_raw.split(",") if x.strip()]
    state_id = STATE_NAME_TO_ABBR.get(state_name, "").lower()

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
            "data": [[price_range, state_id, city.lower().strip(), float(density), category, ingredients, float(coli)]],
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
        if AML_DEPLOYMENT:
            headers["azureml-model-deployment"] = AML_DEPLOYMENT

        with st.spinner("Scoring…"):
            t0 = time.time()
            try:
                resp = requests.post(
                    f"https://{APIM_URL}/aml/score", headers=headers, data=json.dumps(payload), timeout=30
                )
                st.write(f"⏱️ {time.time() - t0:.2f}s")

                ct = resp.headers.get("content-type", "")
                if ct.startswith("application/json"):
                    result = resp.json()
                    if isinstance(result, list) and result:
                        price = result[0]
                    elif isinstance(result, dict):
                        price = result.get("prediction", result.get("value", None))
                    else:
                        price = None

                    if price is not None:
                        st.session_state["prediction"] = float(price)
                    else:
                        st.warning("Prediction received, but couldn't parse price.")
                        st.json(result)
                else:
                    st.code(resp.text, language="json")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.code(json.dumps(payload, indent=2), language="json")

# --- Display result ---
if st.session_state.get("prediction") is not None:
    result_box.success(f"💲 **Predicted price: ${st.session_state['prediction']:,.2f} USD**")
else:
    result_box.empty()
