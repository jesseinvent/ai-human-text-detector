import streamlit as st

from app.predict_text import predict_text

st.set_page_config(page_title="AI vs Human Text Detection", layout="centered")

st.title("AI vs Human Text Detection")

st.write("Enter a text below..")

user_input = st.text_area("Input Text", height=200, placeholder="Type or paste your text here...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        st.spinner("Prediction text...")
        prediction, prediction_prob = predict_text(user_input)
        label = "AI-Generated" if prediction[0] == 'ai' else "Human-Written"
        st.success(f"The text is predicted to be: **{label}**")
        st.write(f"Prediction Probabilities: AI-Generated: {prediction_prob[0][0] * 100:.2f}%, Human-Written: {prediction_prob[0][1] * 100:.2f}%")