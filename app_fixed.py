import streamlit as st
import requests
import json

# Configure the page
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction")
st.markdown("Enter the house details below to get an estimated price.")

# Define input fields matching the FastAPI model
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=1, value=2000, step=10)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)
    stories = st.number_input("Stories", min_value=0, value=2, step=1)
    mainroad = st.selectbox("Main Road Access", options=["no", "yes"], index=0)
    guestroom = st.selectbox("Guest Room", options=["no", "yes"], index=0)
    basement = st.selectbox("Basement", options=["no", "yes"], index=0)

with col2:
    hotwaterheating = st.selectbox("Hot Water Heating", options=["no", "yes"], index=0)
    airconditioning = st.selectbox("Air Conditioning", options=["no", "yes"], index=0)
    parking = st.number_input("Parking Spaces", min_value=0, value=2, step=1)
    prefarea = st.selectbox("Preferred Area", options=["no", "yes"], index=0)
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        options=["unfurnished", "semi-furnished", "furnished"],
        index=0,
    )

# Prepare payload
payload = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus,
}

# Button to trigger prediction
if st.button("Predict Price"):
    # Call the FastAPI endpoint
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            predicted_price = result.get("predicted_price")
            st.success(f"Estimated House Price: **${predicted_price:,.2f}**")
        else:
            st.error(f"API returned error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the FastAPI server is running on http://127.0.0.1:8000")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Optional: Show raw JSON for debugging
with st.expander("Show request payload"):
    st.json(payload)

st.markdown("---")
st.caption("FastAPI backend should be running at http://127.0.0.1:8000")