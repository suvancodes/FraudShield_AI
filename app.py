import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time

# This is the only import you need from your utility file now
from streamlit_utils import load_artifacts, predict_one

# --- Page Configuration ---
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Sample Messages ---
SPAM_EXAMPLE = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"
HAM_EXAMPLE = "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."

# --- Callback Functions to update session state ---
def set_spam_example():
    st.session_state.message_input = SPAM_EXAMPLE

def set_ham_example():
    st.session_state.message_input = HAM_EXAMPLE

# --- Artifact Loading ---
with st.spinner("Loading models and artifacts... Please wait."):
    try:
        preprocessor, w2v_model, model = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

# --- UI Rendering ---
st.title("SMS Spam Detector 🛡️")
st.markdown("Enter a message to classify it as Spam or Ham (not spam). You can also try the examples below.")
st.markdown("---")

# Initialize session state for the text area
if 'message_input' not in st.session_state:
    st.session_state.message_input = ""

# --- Main Application ---
tab1, tab2 = st.tabs(["🔎 Real-time Prediction", "📂 Batch Prediction"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Use session state to manage the text area content
        text_input = st.text_area(
            "Enter Message Text:",
            height=200,
            placeholder="Paste the SMS message here...",
            key="message_input"
        )
        
        # Example buttons with callbacks
        c1, c2 = st.columns(2)
        c1.button("Try Spam Example", on_click=set_spam_example)
        c2.button("Try Ham Example", on_click=set_ham_example)

    with col2:
        st.subheader("Configuration")
        threshold = st.slider("Spam Confidence Threshold", 0.0, 1.0, 0.5, 0.01, help="Adjust the sensitivity of the spam detection. A higher value means the model must be more confident to classify a message as spam.")
        
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Classify Message", type="primary", use_container_width=True, disabled=(not text_input.strip())):
            with st.spinner("Analyzing..."):
                prediction, probability = predict_one(
                    text=text_input,
                    preprocessor=preprocessor,
                    w2v_model=w2v_model,
                    model=model,
                    threshold=threshold
                )
            
            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == "Spam":
                st.error(f"**Result:** {prediction}", icon="🚨")
                st.progress(probability)
                st.write(f"**Confidence:** `{probability:.2%}`")
            elif prediction == "Ham":
                st.success(f"**Result:** {prediction}", icon="✅")
                st.progress(1 - probability) # Show confidence in it being Ham
                st.write(f"**Confidence:** `{1-probability:.2%}`")
            else:
                st.warning("Could not determine prediction. The message may contain no known words.", icon="⚠️")

with tab2:
    st.subheader("Upload a CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="The CSV should have a column named 'message' containing the texts to classify.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "message" not in df.columns:
                st.error("The uploaded CSV must contain a column named 'message'.")
            else:
                st.success(f"File uploaded successfully! Found {len(df)} messages to classify.")
                
                if st.button("Start Batch Prediction", type="primary"):
                    results = []
                    progress_bar = st.progress(0, text="Starting batch prediction...")
                    
                    for i, row in df.iterrows():
                        prediction, probability = predict_one(
                            text=str(row["message"]),
                            preprocessor=preprocessor,
                            w2v_model=w2v_model,
                            model=model,
                            threshold=0.5 # Use a fixed threshold for batch
                        )
                        results.append({
                            "message": str(row["message"]),
                            "prediction": prediction,
                            "spam_probability": f"{probability:.2f}"
                        })
                        progress_bar.progress((i + 1) / len(df), text=f"Processing message {i+1}/{len(df)}")
                    
                    time.sleep(0.5) # For UI to feel complete
                    progress_bar.empty()

                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df)

                    csv_output = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_output,
                        file_name='spam_predictions.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
