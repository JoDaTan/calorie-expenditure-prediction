import streamlit as st
import pandas as pd
import joblib


#---------------------
# Page Setup
#---------------------
st.set_page_config(page_title = 'Calorie Expenditure Predictor')


#---------------------
# Read Data
#---------------------
@st.cache_data
def read_data():
    df = pd.read_csv(r'calories.csv')
    return df

#---------------------
# Load model
#---------------------
@st.cache_resource
def load_model():
    model = joblib.load("calorie-expenditure-predictor")
    return model

data = read_data()
model = load_model()

# -- BMI Calculation --
def calculate_bmi(h, w):
    return w / ((h / 100) ** 2)

st.title("Calorie Expenditure Predictor App")
    
#--------------------
# Input Form 
#--------------------

with st.form(key = "input_form", border = True):
    st.write('Enter bio data')

    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio(
            label = 'Gender',
            options = data['Gender'].unique(),
            horizontal = True
        )
        age = st.slider(
            label = "Age",
            min_value = int(data['Age'].min()),
            max_value = int(data['Age'].max()),
            step = 1
        )
        height = st.number_input(
            label = "Height (cm)",
            min_value = float(data['Height'].min()),
            max_value = float(data['Height'].max()),
            step = 0.1
        )
        weight = st.number_input(
            label = "Weight (kg)",
            min_value = float(data['Weight'].min()),
            max_value = float(data['Weight'].max()),
            step = 0.1
        )

    with col2:
        duration = st.slider(
            label = 'Exercise Duration (minutes)',
            min_value = int(data['Duration'].min()),
            max_value = int(data['Duration'].max()),
            step = 1
        )
        hbpm = st.slider(
            label = "Heart Rate (beats per minutes)",
            min_value = int(data['Heart_Rate'].min()),
            max_value = int(data['Heart_Rate'].max()),
            step = 1
        )
        temperature = st.slider(
            label = "Body Temperature",
            min_value = float(data['Body_Temp'].min()),
            max_value = float(data['Body_Temp'].max()),
            step = 0.1
        )
            

    predict = st.form_submit_button("Predict")

if predict:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Duration": duration,
        "Heart_Rate": hbpm,
        "Body_Temp": temperature,
        "bmi": calculate_bmi(height, weight)
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    with st.container(border = True):
        st.write(f'Estimated Calories Burned: {prediction:.2f}')