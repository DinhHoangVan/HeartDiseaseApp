import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

st.write("""
# Heart Failure Prediction App

""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe

def user_input_features():
    model = st.sidebar.selectbox('Model', ('Logistic Regression', 'SVC', 'XGBoost'))


    Age = st.sidebar.slider('Age', 20, 90, 70)
    Sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    ChestPainType = st.sidebar.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymtomatic'))
    RestingBP = st.sidebar.slider('Resting systolic blood pressure(mmHg)', 94, 200, 160)
    Cholesterol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 85, 564, 140)
    FastingBS = st.sidebar.selectbox('Fasting Blood Sugar', ('Greater than 120 mg/dl', 'Less than 120 mg/dl'))
    RestingECG = st.sidebar.selectbox('Resting ECG Results', ('Normal', 'Having ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'))
    MaxHR = st.sidebar.slider('Max Heart Rate (bpm)', 71, 202, 160)
    ExerciseAngina = st.sidebar.selectbox('Exercise-induced Angina', ('Yes', 'No'))
    Oldpeak = st.sidebar.slider('ST depression', .0,7.0,1.1)
    ST_Slope = st.sidebar.selectbox('slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))



    data = {'Age': Age,  #
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,  #
        'Cholesterol': Cholesterol,  #
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,  #
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
        }

    features = pd.DataFrame(data, index=[0])
    return features, model
df, model = user_input_features()
# Encode inputs
encode = ['ExerciseAngina']
cat_mapping = {'Yes': 1, 'No': 0 }
for col in encode:
    df[col] = df[col].map(cat_mapping)
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['ChestPainType'] = df['ChestPainType'].map(
    {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymtomatic': 3})
df['FastingBS'] = df['FastingBS'].map({'Greater than 120 mg/dl': 1, 'Less than 120 mg/dl': 0})
df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'Having ST-T wave abnormality': 1, 'Probable or definite left ventricular hypertrophy': 2})
df['ST_Slope'] = df['ST_Slope'].map({'Upsloping': 0, 'Flat': 1, 'Downsloping': 2})
# Displays the user input features
st.subheader('User Input features')

st.write(df.iloc[:, :11])
#st.write(df.iloc[:, 6:])

# Save column names for later
columns_list = list(df.columns)

# Reading the original dataset in to scale the input
HF_train = pd.read_csv('heart.csv')
HF_train = HF_train.drop('HeartDisease', axis=1)

df = pd.concat([df, HF_train], axis=0)

scale_list = {'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG','MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope' }
scaler = StandardScaler()
df[list(scale_list)] = scaler.fit_transform(df[list(scale_list)])
# Getting back just the user input
df = df[:1]

# Load in trained models

if model == 'Logistic Regression':
    load_clf = pickle.load(open('Logistic_HF.pkl', 'rb'))
elif model == 'SVC':
    load_clf = pickle.load(open('SVC_HF.pkl', 'rb'))
elif model == 'XGBoost':
    load_clf = pickle.load(open('XGB_HF.pkl', 'rb'))

prediction = load_clf.predict(df)

# Prediction Probabilities

st.write('#')
st.subheader('Heart Failure Prediction')
HeartDisease = np.array(['NO', 'YES'])
st.write(HeartDisease[prediction])

prediction_proba = load_clf.predict_proba(df)
prediction_proba = np.round(prediction_proba, 4)
proba_df = pd.DataFrame(prediction_proba)

print(proba_df)

st.subheader('Heart Disease Probabilities')
st.dataframe(proba_df.style.format("{:.2%}"))
# Feature importances

if model == 'XGBoost':

    st.subheader('XGB Feature Importance')

    imp_df = pd.DataFrame(list(zip(columns_list, load_clf.feature_importances_)), columns=['Feature', 'Importance'])

    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x="Feature", y="Importance", data=imp_df)
    st.pyplot(fig)

    st.write(imp_df)

elif model == 'Logistic Regression':
    st.subheader('Regression Coefficients')

    imp_df = pd.DataFrame(list(zip(columns_list, load_clf.coef_[0])), columns=['Feature', 'Coefficient'])

    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x="Feature", y="Coefficient", data=imp_df)
    st.pyplot(fig)

    st.write(imp_df)

elif model == 'SVC':
    st.subheader('SVC Coefficients')

    imp_df = pd.DataFrame(list(zip(columns_list, load_clf.coef_[0])), columns=['Feature', 'Coefficient'])

    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x="Feature", y="Coefficient", data=imp_df)
    st.pyplot(fig)

    st.write(imp_df)


