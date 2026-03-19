import streamlit as st
import requests

st.set_page_config(page_title="Phishing Detection System")

st.title("Phishing Website Detection")
st.write("Enter a URL and select a model to analyze the website.")

url = st.text_input("Enter Website URL")

model_name = st.selectbox(
    "Select Model",
    ["AdaptiveCNN", "BaselineCNN"]
)

BACKEND_URL = "http://127.0.0.1:8000"

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "feedback_sent" not in st.session_state:
    st.session_state["feedback_sent"] = {"false_alarm": False, "missed_phishing": False}

if st.button("Analyze"):

    if url.strip() == "":
        st.warning("Please enter a valid URL")
    else:
        with st.spinner("Analyzing..."):

            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={
                        "url": url,
                        "model_name": model_name
                    }
                )
                response.raise_for_status()

                result = response.json()
                st.session_state["last_result"] = result
                st.session_state["feedback_sent"] = {"false_alarm": False, "missed_phishing": False}

            except Exception as e:
                st.error(f"Error: {e}")

result = st.session_state.get("last_result")
if result:
    st.subheader("Results")

    st.write("**Model Used:**", result.get("model_used"))
    st.write("**Prediction:**", result.get("prediction"))
    st.write("**Confidence:**", result.get("confidence"))

    if result.get("prediction") == "Phishing":
        st.error("This website is likely PHISHING.")
    else:
        st.success("This website appears LEGITIMATE.")

    st.subheader("Feedback")
    feedback_notice = st.empty()

    # Make the buttons sit closer together
    btn_col1, btn_col2, _spacer = st.columns([1, 1, 1])

    with btn_col1:
        if st.button(
            "Report False Alarm",
            disabled=st.session_state["feedback_sent"]["false_alarm"],
            key="report_false_alarm_btn",
        ):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/report_false_alarm",
                    json={"url": result["url"], "model_name": result["model_used"]},
                    timeout=10,
                )
                r.raise_for_status()
                st.session_state["feedback_sent"]["false_alarm"] = True
                feedback_notice.success("False prediction reported")
            except Exception as e:
                feedback_notice.error(f"Failed to submit feedback: {e}")

    with btn_col2:
        if st.button(
            "Report Missed Phishing",
            disabled=st.session_state["feedback_sent"]["missed_phishing"],
            key="report_missed_phishing_btn",
        ):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/report_missed_phishing",
                    json={"url": result["url"], "model_name": result["model_used"]},
                    timeout=10,
                )
                r.raise_for_status()
                st.session_state["feedback_sent"]["missed_phishing"] = True
                feedback_notice.success("False prediction reported")
            except Exception as e:
                feedback_notice.error(f"Failed to submit feedback: {e}")
