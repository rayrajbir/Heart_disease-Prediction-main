import streamlit as st
import pandas as pd
import pickle

# App Title and Intro
st.markdown("""
    <div style="background-color:tomato;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center"> 👩‍⚕️Heart Disease Prediction 🧑‍⚕️</h1>
    </div>
    """, unsafe_allow_html=True)

st.write("""
    #### Welcome to the Heart Disease Prediction App!  
    **Follow these steps:**
    - **1.** Enter your health details in the sidebar.
    - **2.** Press the "Predict" button below to check your result!
""")

# Sidebar input
st.title('🔍 Answer the following health questions:')

BMI = st.selectbox("What is your BMI?", 
                           ("Normal weight BMI (18.5-25)", 
                            "Under-weight BMI (< 18.5)", 
                            "Over-weight BMI (25-30)", 
                            "Obese BMI (> 30)"))

Age = st.selectbox("Select your age range", 
                           ("18-24", "25-29", "30-34", "35-39", 
                            "40-44", "45-49", "50-54", "55-59", 
                            "60-64", "65-69", "70-74", "75-79", 
                            "80 or older"))

Race = st.selectbox("Select your race", 
                            ("Asian", "Black", "Hispanic", 
                             "American Indian/Alaskan Native", 
                             "White", "Other"))

Gender = st.selectbox("Enter your gender", 
                              ("Female", "Male"))

Smoking = st.selectbox("Have you smoked more than 100 cigarettes in your entire life?", 
                               ("No", "Yes"))

alcoholDrink = st.selectbox("Do you drink alcohol often?", 
                                    ("No", "Yes"))

stroke = st.selectbox("Have you ever had a stroke?", 
                              ("No", "Yes"))

sleepTime = st.number_input("How many hours do you sleep per day?", 0, 24, 7)

genHealth = st.selectbox("How would you rate your health?", 
                                 ("Excellent", "Very good", "Good", "Fair", "Poor"))

physHealth = st.number_input("Physical Health Score (0 for Excellent, for Very bad)", 0, 30, 0)

mentHealth = st .number_input("Mental Health Score (0 for Excellent, 30 for Very bad)", 0, 30, 0)

physAct = st.selectbox("Do you exercise often?", 
                               ("No", "Yes"))

diffWalk = st.selectbox("Do you have difficulty walking or climbing stairs?", 
                                ("No", "Yes"))

diabetic = st.selectbox("Have you ever had diabetes?", 
                                ("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))

asthma = st .selectbox("Have you ever had asthma?", 
                              ("No", "Yes"))

kidneyDisease = st .selectbox("Have you ever had kidney disease?", 
                                     ("No", "Yes"))

skinCancer = st .selectbox("Have you ever had skin cancer?", 
                                  ("No", "Yes"))

# Prepare data for prediction
dataToPredic = pd.DataFrame({
    "BMI": [BMI],
    "Smoking": [Smoking],
    "AlcoholDrinking": [alcoholDrink],
    "Stroke": [stroke],
    "PhysicalHealth": [physHealth],
    "MentalHealth": [mentHealth],
    "DiffWalking": [diffWalk],
    "Sex": [Gender],
    "AgeCategory": [Age],
    "Race": [Race],
    "Diabetic": [diabetic],
    "PhysicalActivity": [physAct],
    "GenHealth": [genHealth],
    "SleepTime": [sleepTime],
    "Asthma": [asthma],
    "KidneyDisease": [kidneyDisease],
    "SkinCancer": [skinCancer]
})

# Mapping categorical values to numerical values
replace_dict = {
    "BMI": {
        "Under-weight BMI (< 18.5)": 0,
        "Normal weight BMI (18.5-25)": 1,
        "Over-weight BMI (25-30)": 2,
        "Obese BMI (> 30)": 3
    },
    "AgeCategory": {
        "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4,
        "45-49": 5, "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9,
        "70-74": 10, "75-79": 11, "80 or older": 12
    },
    "Diabetic": {
        "No": 0, "Yes": 1, "Yes, during pregnancy": 2, "No, borderline diabetes": 3
    },
    "GenHealth": {
        "Excellent": 0, "Very good": 1, "Good": 2, "Fair": 3, "Poor": 4
    },
    "Race": {
        "White": 0, "Other": 1, "Black": 2, "Hispanic": 3, "Asian": 4,
        "American Indian/Alaskan Native": 5
    },
    "Sex": {
        "Female": 0, "Male": 1
    },
    "Smoking": {"No": 0, "Yes": 1},
    "AlcoholDrinking": {"No": 0, "Yes": 1},
    "Stroke": {"No": 0, "Yes": 1},
    "DiffWalking": {"No": 0, "Yes": 1},
    "PhysicalActivity": {"No": 0, "Yes": 1},
    "Asthma": {"No": 0, "Yes": 1},
    "KidneyDisease": {"No": 0, "Yes": 1},
    "SkinCancer": {"No": 0, "Yes": 1}
}

dataToPredic.replace(replace_dict, inplace=True)

# Load the saved machine learning model
filename = 'LogRegModel.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Prediction
Result = loaded_model.predict(dataToPredic)
ResultProb = loaded_model.predict_proba(dataToPredic)

# Prediction button
if st.button('🚀 PREDICT'):
    if ResultProb[0][1] >= 0.15:
        st.markdown(f"<h3 style='color:red;'>🔴 Prediction: Yes, you may have heart disease.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:green;'>🟢 Prediction: No, you are less likely to have heart disease.</h3>", unsafe_allow_html=True)
    
    st.write(f'🔬 **Chance of heart disease**: {ResultProb[0][1] * 100:.2f}%')



