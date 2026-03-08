import streamlit as st
import requests

st.set_page_config(page_title="Phishing Detection System")

st.title("Phishing Website Detection")
st.write("Enter a URL and select a model to analyze the website.")

# URL input
url = st.text_input("Enter Website URL")

# Model selection
model_name = st.selectbox(
    "Select Model",
    ["AdaptiveCNN", "BaselineCNN"]
)

if st.button("Analyze"):

    if url.strip() == "":
        st.warning("Please enter a valid URL")
    else:
        with st.spinner("Analyzing..."):

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={
                        "url": url,
                        "model_name": model_name
                    }
                )

                result = response.json()

                st.subheader("Results")

                st.write("**URL:**", result["url"])
                st.write("**Model Used:**", result["model_used"])
                st.write("**Prediction:**", result["prediction"])
                st.write("**Confidence:**", result["confidence"])

                if result["prediction"] == "Phishing":
                    st.error("This website is likely PHISHING.")
                else:
                    st.success("This website appears LEGITIMATE.")

            except Exception as e:
                st.error(f"Error: {e}")