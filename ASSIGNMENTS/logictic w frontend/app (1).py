import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("titanic_model.pkl", "rb"))
scaler = pickle.load(open("titanic_scaler.pkl", "rb"))

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict if they would have survived.")


Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ['male', 'female'])
Age = st.number_input("Age", 0, 100, step=1)
SibSp = st.number_input("Number of Siblings/Spouse aboard", 0, 10)
Parch = st.number_input("Number of Parents/Children aboard", 0, 10)
Fare = st.number_input("Fare", 0.0, 600.0)
Embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Encoding
Sex = 1 if Sex == 'male' else 0
Embarked = {'S': 2, 'C': 0, 'Q': 1}[Embarked]

# Predict
if st.button("Predict"):
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)[0]

    if result == 1:
        st.success("🟢 Survived")
    else:
        st.error("🔴 Did Not Survive")
