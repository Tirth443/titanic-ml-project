import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model (Pipeline)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
    st.write("Model type:", type(model).__name__)
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Powered by trained GradientBoostingClassifier (Accuracy ~82%)")

# Exact feature columns from preprocessing (order critical!)
FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'FarePerPerson', 'Title', 'AgeGroup']

# User inputs
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.slider("SibSp", 0, 8, 0)
    parch = st.slider("Parch", 0, 6, 0)
with col2:
    fare = st.number_input("Fare", 0.0, 500.0, 20.0)
    embarked = st.selectbox("Embarked", ['S', 'C', 'Q'], index=0)

# Derived features (match preprocessing exactly)
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_per_person = fare / family_size if family_size > 0 else fare

# AgeGroup match pd.cut
age_group = pd.cut([age], bins=[0,12,20,40,60,80], labels=[0,1,2,3,4], include_lowest=True)[0]
if pd.isna(age_group):
    age_group = 2.0

# Title hardcoded demo (Mr=0)
title = 0

if st.button("🎯 Predict Survival", type="primary"):
    # Encoding match preprocessing
    sex_code = 0 if sex == "male" else 1
    embarked_code = {'S':0, 'C':1, 'Q':2}[embarked]

    # DataFrame with EXACT column order
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_code],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked_code],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'FarePerPerson': [fare_per_person],
        'Title': [title],
        'AgeGroup': [age_group]
    })

    # Reorder to match training order
    data = data[FEATURE_COLS]

    st.write("**Input features:**")
    st.dataframe(data)

    try:
        # Predict
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0]

        st.subheader("**Prediction:**")
        if prediction == 1:
            st.success("🎉 **Survived**")
        else:
            st.error("❌ **Did Not Survive**")

        # Probability bar
        st.subheader("**Survival Probability:**")
        st.progress(proba[1])
        col_prob1, col_prob0 = st.columns(2)
        with col_prob1:
            st.metric("Survived", f"{proba[1]:.1%}")
        with col_prob0:
            st.metric("Died", f"{proba[0]:.1%}")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write("Debug: data.shape", data.shape, "columns:", list(data.columns))

st.info("💡 Model trained on Titanic data. Try different inputs!")
