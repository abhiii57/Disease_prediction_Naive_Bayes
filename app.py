import streamlit as st
import joblib
import pandas as pd

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🧠",
    layout="centered",
)

# -------------------- Cache Model & Data --------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("Predict_disease.joblib")

@st.cache_data(show_spinner=False)
def load_symptoms():
    data = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    X = data.drop("diseases", axis=1)
    return X.columns.tolist()

model = load_model()
all_symptoms = load_symptoms()

# -------------------- Initialize Session State --------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -------------------- App UI --------------------
st.title("🩺 Disease Prediction ")
st.markdown(
    """
    Enter your **symptoms** separated by commas (e.g.  
    `fever, cough, fatigue`) and get an instant prediction  
    based on our trained machine learning model.
    """
)

user_input = st.text_input(
    "Enter Symptoms:",
    placeholder="e.g. shortness of breath, chest pain, fatigue",
    key="symptoms_input"
)

# -------------------- Predict Button --------------------
def predict_disease():
    user_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]
    input_vector = pd.DataFrame(
        [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
        columns=all_symptoms
    )
    st.session_state.prediction = model.predict(input_vector)[0]

if st.button("Predict Disease"):
    if not user_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        predict_disease()

# -------------------- Display Prediction --------------------
if st.session_state.prediction:
    st.success(f"🧬 **Predicted Disease:** {st.session_state.prediction}")
    st.markdown("---")
    st.markdown("### Note")
    st.info(
        "This prediction is based on symptom patterns from our dataset. "
        "Consult a medical professional for accurate diagnosis and treatment."
    )

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed using Streamlit.")
