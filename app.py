import streamlit as st
import joblib
import pandas as pd
st.set_page_config(page_title="Disease Predictor", page_icon="🧠", layout="centered")

# -------------------- Load model and data --------------------
@st.cache_resource
def load_model_and_data():
    model = joblib.load("Predict_disease.joblib")
    data = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    X = data.drop("diseases", axis=1)
    all_symptoms = X.columns.tolist()
    return model, all_symptoms

model, all_symptoms = load_model_and_data()

# -------------------- App UI --------------------

st.title("🩺 Disease Prediction App")
st.markdown(
    """
    Enter your **symptoms** separated by commas (e.g.,  
    `fever, cough, fatigue`) and get an instant prediction  
    based on your trained machine learning model.
    """
)

user_input = st.text_input("Enter Symptoms:", placeholder="e.g. shortness of breath, chest pain, fatigue")

# -------------------- Predict Button --------------------
if st.button("Predict Disease"):
    if not user_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        user_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]
        input_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )

        # Make prediction
        result = model.predict(input_vector)[0]

        # Display result
        st.success(f"🧬 **Predicted Disease:** {result}")

        # Optional extra info section
        st.markdown("---")
        st.markdown("### 🩹 Tips")
        st.info(
            "This prediction is based on symptom patterns from our dataset. "
            "Consult a medical professional for accurate diagnosis and treatment."
        )

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit.")
