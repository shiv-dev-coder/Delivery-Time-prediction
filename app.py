
import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Delivery Predictor",
    page_icon="🚚",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("food_delivery_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

FEATURE_COLUMNS = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

CATEGORY_MAPPINGS = {
    "Weather": {"Clear": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3, "Windy": 4},
    "Traffic_Level": {"High": 0, "Low": 1, "Medium": 2},
    "Time_of_Day": {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3},
    "Vehicle_Type": {"Bike": 0, "Car": 1, "Scooter": 2},
}

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
        cursor: pointer;
    }

    .stApp {
        background: linear-gradient(135deg, #050816, #0f172a, #111827);
        color: white;
    }

    /* Animated Background */
    .animated-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: -1;
    }

    .circle {
        position: absolute;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
        animation: float 10s infinite ease-in-out;
    }

    .circle:nth-child(1) {
        width: 250px;
        height: 250px;
        top: 10%;
        left: 10%;
    }

    .circle:nth-child(2) {
        width: 300px;
        height: 300px;
        bottom: 10%;
        right: 10%;
        animation-delay: 2s;
    }

    .circle:nth-child(3) {
        width: 180px;
        height: 180px;
        bottom: 30%;
        left: 40%;
        animation-delay: 4s;
    }

    @keyframes float {
        0% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-30px);
        }
        100% {
            transform: translateY(0px);
        }
    }

    /* Main Container */
    .glass {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
        padding: 35px;
        border-radius: 25px;
        margin-top: 20px;
    }

    .title {
        font-size: 52px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(to right, #60a5fa, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #d1d5db;
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #9333ea, #ec4899);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        cursor: pointer;
    }

    /* Result Card */
    .result-box {
        background: rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 30px;
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeIn 0.8s ease-in-out;
    }

    .result-text {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0px);
        }
    }

    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- BACKGROUND ----------------
st.markdown(
    """
    <div class="animated-bg">
        <div class="circle"></div>
        <div class="circle"></div>
        <div class="circle"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🚚 Food Delivery Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Random Forest Machine Learning Model</div>', unsafe_allow_html=True)

# ---------------- FORM ----------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    distance_km = st.slider(
        "Distance (KM)",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.1
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
    )

    traffic = st.selectbox(
        "Traffic Level",
        ["Low", "Medium", "High"]
    )

with col2:
    time_of_day = st.selectbox(
        "Time Of Day",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        ["Bike", "Car", "Scooter"]
    )

    preparation_time = st.slider(
        "Preparation Time (Minutes)",
        min_value=1,
        max_value=60,
        value=15
    )

    courier_experience = st.slider(
        "Courier Experience (Years)",
        min_value=0.0,
        max_value=15.0,
        value=2.0,
        step=0.5
    )

predict_btn = st.button("✨ Predict Delivery Time")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:

    # Create dataframe with the same label encoding used during training.
    input_data = pd.DataFrame({
        'Distance_km': [distance_km],
        'Weather': [CATEGORY_MAPPINGS["Weather"][weather]],
        'Traffic_Level': [CATEGORY_MAPPINGS["Traffic_Level"][traffic]],
        'Time_of_Day': [CATEGORY_MAPPINGS["Time_of_Day"][time_of_day]],
        'Vehicle_Type': [CATEGORY_MAPPINGS["Vehicle_Type"][vehicle]],
        'Preparation_Time_min': [preparation_time],
        'Courier_Experience_yrs': [courier_experience]
    })
    input_data = input_data[FEATURE_COLUMNS]

    # Prediction
    prediction = model.predict(input_data)[0]

    # Success Animation
    st.balloons()

    # Result Card
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-text">
                ⏱ Estimated Delivery Time<br><br>
                {round(prediction, 2)} Minutes
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#9ca3af;'>
        Built with Streamlit + GSAP + Random Forest 🚀
    </div>
    """,
    unsafe_allow_html=True
)
