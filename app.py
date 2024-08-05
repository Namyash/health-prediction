import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib

# Load models and scalers
lungmodel = joblib.load('logistic_regression_lung_model.pkl')
heartmodel = joblib.load('neural_network_heart_model.pkl')
dbmodel = joblib.load('random_forest_diabetes_model.pkl')

heart_scaler_input = joblib.load('heart_scaler.pkl')
lung_scaler_input = joblib.load('lung_scaler.pkl')
db_scaler_input = joblib.load('db_scaler.pkl')

def main():
    st.markdown(
        """
        <style>
        body {
            background-color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        
    st.title("Health Prediction Models")
    
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Heart Attack", "Lung Cancer", "Diabetes"],
            icons=["house", "activity", "lungs", "droplet"],
            menu_icon="cast",
            default_index=0,
        )
    
    if selected == "Home":
        st.header("Welcome to the Health Prediction Models App")
        st.write("Please select a prediction model or read about health conditions from the sidebar.")
    
    elif selected == "Heart Attack":
        sub_selected = option_menu(
            menu_title="Heart Attack Menu",
            options=["Heart Attack Prediction", "About Heart Attack"],
            icons=["activity", "book"],
            menu_icon="cast",
            default_index=0,
        )
        
        if sub_selected == "Heart Attack Prediction":
            st.header("Heart Attack Prediction")
            with st.container():
                st.subheader("Patient Information")
                
                row1 = st.columns(4)
                with row1[0]:
                    age = st.number_input('Age', value=50.0)
                with row1[1]:
                    sex = st.selectbox('Sex', ['Female', 'Male'])
                with row1[2]:
                    cp = st.number_input('Chest Pain Type', value=1.0)
                with row1[3]:
                    trtbps = st.number_input('Resting Blood Pressure', value=120.0)
                
                row2 = st.columns(4)
                with row2[0]:
                    chol = st.number_input('Cholesterol', value=200.0)
                with row2[1]:
                    fbs = st.selectbox('Fasting Blood Sugar', ['No', 'Yes'])
                with row2[2]:
                    restecg = st.number_input('Rest ECG', value=0.0)
                with row2[3]:
                    thalachh = st.number_input('Max. HeartRate Achieved', value=150.0)
                
                row3 = st.columns(4)
                with row3[0]:
                    exng = st.selectbox('Exercise Induced Angina', [0.0, 1.0])
                with row3[1]:
                    oldpeak = st.number_input('Oldpeak', value=1.0)
                with row3[2]:
                    slp = st.selectbox('SLP', [0.0, 1.0, 2.0])
                with row3[3]:
                    caa = st.selectbox('No. Vessels Colored FLS', [0.0, 1.0, 2.0, 3.0])
                
                row4 = st.columns(4)
                with row4[0]:
                    thall = st.selectbox('Thalassemia', [0.0, 1.0, 2.0, 3.0])
            input_data = np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]).reshape(1, -1)
            try:
                input_data_scaled = heart_scaler_input.transform(input_data)
            except Exception as e:
                st.error(f"Scaling error: {e}")
                return
            if st.button('Predict Heart Attack'):
                try:
                    prediction = heartmodel.predict(input_data_scaled)
                    st.write(f'Heart Attack Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        elif sub_selected == "About Heart Attack":
            st.header("Heart Attack: Causes and Prevention")
            st.write("""
                ## Causes of Heart Attack
                - **High Blood Pressure**: Damages arteries and leads to heart disease.
                - **High Cholesterol**: Can clog arteries and lead to heart attacks.
                - **Smoking**: Increases the risk of heart disease.
                - **Obesity**: Excess weight puts strain on the heart.
                - **Physical Inactivity**: Lack of exercise increases the risk of heart disease.
                - **Poor Diet**: High in saturated fats, trans fats, and cholesterol.
                
                ## Prevention
                - **Healthy Diet**: Focus on fruits, vegetables, and whole grains.
                - **Regular Exercise**: At least 150 minutes of moderate aerobic activity per week.
                - **Weight Management**: Maintain a healthy weight.
                - **Avoid Smoking**: Quit smoking and avoid secondhand smoke.
                - **Control Blood Pressure and Cholesterol**: Regular check-ups and medications if necessary.
                - **Stress Management**: Practice relaxation techniques and seek support when needed.
            """)

    elif selected == "Lung Cancer":
        sub_selected = option_menu(
            menu_title="Lung Cancer Menu",
            options=["Lung Cancer Prediction", "About Lung Cancer"],
            icons=["lungs", "book"],
            menu_icon="cast",
            default_index=0,
        )
        
        if sub_selected == "Lung Cancer Prediction":
            st.header("Lung Cancer Prediction")
            with st.container():
                st.subheader("Patient Information")
                row1 = st.columns(4)
                with row1[0]:
                    age = st.number_input('Age', value=50.0)
                    yellow_fingers = st.selectbox('Yellow Fingers', [0, 1])
                    peer_pressure = st.selectbox('Peer Pressure', [0, 1])
                    fatigue = st.selectbox('Fatigue', [0, 1])
                with row1[1]:
                    wheezing = st.selectbox('Wheezing', [0, 1])
                    coughing = st.selectbox('Coughing', [0, 1])
                    swallowing_difficulty = st.selectbox('Swallowing Difficulty', [0, 1])
                    smoking = st.selectbox('Smoking', [0, 1])
                with row1[2]:
                    anxiety = st.selectbox('Anxiety', [0, 1])
                    chronic_disease = st.selectbox('Chronic Disease', [0, 1])
                    allergy = st.selectbox('Allergy', [0, 1])
                    alcohol_consuming = st.selectbox('Alcohol Consuming', [0, 1])
                with row1[3]:
                    shortness_of_breath = st.selectbox('Shortness of Breath', [0, 1])
                    chest_pain = st.selectbox('Chest Pain', [0, 1])
            input_data = np.array([age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]).reshape(1, -1)
            try:
                input_data_scaled = lung_scaler_input.transform(input_data)
            except Exception as e:
                st.error(f"Scaling error: {e}")
                return
            if st.button('Predict Lung Cancer'):
                try:
                    prediction = lungmodel.predict(input_data_scaled)
                    st.write(f'Lung Cancer Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        elif sub_selected == "About Lung Cancer":
            st.header("Lung Cancer: Causes and Prevention")
            st.write("""
                ## Causes of Lung Cancer
                - **Smoking**: The leading cause of lung cancer.
                - **Secondhand Smoke**: Inhalation of smoke from other people's cigarettes.
                - **Exposure to Radon Gas**: A naturally occurring radioactive gas.
                - **Exposure to Asbestos and Other Carcinogens**: Workplace exposure to harmful substances.
                - **Family History**: Genetics can play a role.
                
                ## Prevention
                - **Avoid Smoking**: Quit smoking and avoid secondhand smoke.
                - **Test Your Home for Radon**: Mitigate if necessary.
                - **Avoid Carcinogens at Work**: Follow safety protocols to reduce exposure.
                - **Eat a Healthy Diet**: Include plenty of fruits and vegetables.
                - **Exercise Regularly**: Physical activity can help lower the risk of lung cancer.
                - **Prevent Infections**: Regular vaccinations can help reduce the risk.
            """)

    elif selected == "Diabetes":
        sub_selected = option_menu(
            menu_title="Diabetes Menu",
            options=["Diabetes Prediction", "About Diabetes"],
            icons=["droplet", "book"],
            menu_icon="cast",
            default_index=0,
        )
        
        if sub_selected == "Diabetes Prediction":
            st.header("Diabetes Prediction")
            with st.container():
                st.subheader("Patient Information")
                row1 = st.columns(4)
                with row1[0]:
                    pregnancies = st.number_input('Pregnancies', value=0)
                    glucose = st.number_input('Glucose', value=100)
                    skin_thickness = st.number_input('Skin Thickness', value=20)
                    bmi = st.number_input('BMI', value=25.0)
                with row1[1]:
                    blood_pressure = st.number_input('Blood Pressure', value=70)
                    insulin = st.number_input('Insulin', value=79)
                    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', value=0.5)
                    age = st.number_input('Age', value=33)
                input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
            try:
                input_data_scaled = db_scaler_input.transform(input_data)
            except Exception as e:
                st.error(f"Scaling error: {e}")
                return
            if st.button('Predict Diabetes'):
                try:
                    prediction = dbmodel.predict(input_data_scaled)
                    st.write(f'Diabetes Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        elif sub_selected == "About Diabetes":
            st.header("Diabetes: Causes and Prevention")
            st.write("""
                ## Causes of Diabetes
                - **Genetics**: Family history can play a significant role.
                - **Lifestyle Factors**: Poor diet, lack of exercise, and obesity.
                - **Age**: Risk increases with age.
                - **High Blood Pressure**: Can contribute to the development of diabetes.
                - **Insulin Resistance**: Cells become resistant to the action of insulin.
                
                ## Prevention
                - **Healthy Diet**: Focus on fruits, vegetables, and whole grains.
                - **Regular Exercise**: Helps control weight and lowers blood sugar levels.
                - **Weight Management**: Maintain a healthy weight.
                - **Regular Monitoring**: Keep track of blood sugar levels.
                - **Avoid Smoking**: Smoking increases the risk of diabetes and cardiovascular disease.
                - **Stress Management**: Practice relaxation techniques and seek support when needed.
            """)

if __name__ == '__main__':
    main()
