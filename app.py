import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the pre-trained models and columns
best_logreg = joblib.load('best_logreg_model.pkl')
best_rf = joblib.load('best_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. Define constants for imputation and feature engineering
AGE_MEDIAN = 28.0 # Global median from training data
FARE_MEDIAN = 14.4542 # Global median from training data
EMBARKED_MODE = 'S' # Global mode from training data

FARE_BINS = [0.0, 7.9104, 14.4542, 31.0, 512.3292] # Adjusted based on qcut boundaries from training
FARE_LABELS = ['Low', 'Medium', 'High', 'Very High']

# Title mapping from feature_engineering function
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Rare', 'Sir': 'Rare',
    'Mme': 'Mrs', 'Don': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare'
}

def process_user_input(input_data):
    # Convert input_data dictionary to a DataFrame
    df_user = pd.DataFrame([input_data])

    # Ensure column types are consistent with original df for processing
    # For simplicity, PassengerId, Ticket are not used and Cabin is handled specially.
    # Name is used only for title extraction.
    df_user['Name'] = df_user['Name'].fillna('Mr. User') # Dummy name for title extraction if not provided
    df_user['Cabin'] = df_user['Cabin'].replace('', np.nan) # Treat empty string as NaN for cabin

    # Extract Title
    df_user['Title'] = df_user['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df_user['Title'] = df_user['Title'].map(title_mapping).fillna('Rare') # Handle unseen titles

    # Family size features
    df_user['FamilySize'] = df_user['SibSp'] + df_user['Parch'] + 1
    df_user['IsAlone'] = (df_user['FamilySize'] == 1).astype(int)

    # Cabin features
    df_user['Deck'] = df_user['Cabin'].str[0]
    df_user['Deck'] = df_user['Deck'].fillna('Unknown')
    df_user['HasCabin'] = df_user['Cabin'].notna().astype(int)

    # Age - fill with global median
    df_user['Age'] = df_user['Age'].fillna(AGE_MEDIAN)
    df_user['AgeGroup'] = pd.cut(df_user['Age'],
                                 bins=[0, 12, 18, 35, 60, 100],
                                 labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'],
                                 right=True)
    # Ensure 'AgeGroup' is categorical with all possible categories
    age_group_categories = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
    df_user['AgeGroup'] = pd.Categorical(df_user['AgeGroup'], categories=age_group_categories)

    # Fare - fill with global median
    df_user['Fare'] = df_user['Fare'].fillna(FARE_MEDIAN)
    df_user['FareGroup'] = pd.cut(df_user['Fare'],
                                  bins=FARE_BINS,
                                  labels=FARE_LABELS,
                                  include_lowest=True)
    # Ensure 'FareGroup' is categorical with all possible categories
    fare_group_categories = ['Low', 'Medium', 'High', 'Very High']
    df_user['FareGroup'] = pd.Categorical(df_user['FareGroup'], categories=fare_group_categories)

    # Embarked - fill with mode
    df_user['Embarked'] = df_user['Embarked'].fillna(EMBARKED_MODE)

    # Select and drop columns to match the df_clean state from training
    cols_to_keep = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'HasCabin', 'AgeGroup', 'FareGroup']
    df_processed = df_user[cols_to_keep]

    # One-hot encode categorical variables
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

    # Align columns with the training data's model_columns
    # Add missing columns (which would be False/0 for this single instance)
    missing_cols = set(model_columns) - set(df_encoded.columns)
    for c in missing_cols:
        df_encoded[c] = False # Use False for boolean dtypes from get_dummies

    # Drop any extra columns that might have appeared (shouldn't happen if model_columns is comprehensive)
    extra_cols = set(df_encoded.columns) - set(model_columns)
    df_encoded = df_encoded.drop(columns=list(extra_cols))

    # Reorder columns to match the training data
    df_final = df_encoded[model_columns]

    return df_final

# Streamlit app layout
st.title('Titanic Survival Predictor')
st.write('Enter passenger details to predict survival probability.')

# User input widgets
st.sidebar.header('Passenger Details')
pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
sex = st.sidebar.radio('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0, 80, 30)
sibsp = st.sidebar.slider('SibSp (Siblings/Spouses)', 0, 8, 0)
parch = st.sidebar.slider('Parch (Parents/Children)', 0, 6, 0)
fare = st.sidebar.number_input('Fare', min_value=0.0, max_value=500.0, value=30.0)
cabin = st.sidebar.text_input('Cabin (e.g., C23, leave blank if unknown)', '')
embarked = st.sidebar.selectbox('Embarked', ['S', 'C', 'Q'])

# Store user inputs in a dictionary
user_input = {
    'PassengerId': 0, # Dummy for preprocessing, not used in model directly
    'Pclass': pclass,
    'Name': 'Mr. User' if sex == 'male' else 'Mrs. User', # Dummy name for title extraction
    'Sex': sex,
    'Age': float(age),
    'SibSp': sibsp,
    'Parch': parch,
    'Ticket': '0', # Dummy for preprocessing, not used in model directly
    'Fare': float(fare),
    'Cabin': cabin,
    'Embarked': embarked
}

if st.sidebar.button('Predict Survival'):
    # Process user input
    processed_input = process_user_input(user_input)

    st.subheader('Prediction Results')

    # Logistic Regression Prediction
    logreg_prediction_proba = best_logreg.predict_proba(processed_input)
    logreg_survived_proba = logreg_prediction_proba[0][1] # Probability of surviving
    logreg_prediction = best_logreg.predict(processed_input)[0]

    st.write('### Logistic Regression Model')
    if logreg_prediction == 1:
        st.success(f'Prediction: \u2705 Survived (Probability: {logreg_survived_proba:.2f})')
    else:
        st.error(f'Prediction: \u274C Not Survived (Probability: {1-logreg_survived_proba:.2f})')

    # Random Forest Prediction
    rf_prediction_proba = best_rf.predict_proba(processed_input)
    rf_survived_proba = rf_prediction_proba[0][1] # Probability of surviving
    rf_prediction = best_rf.predict(processed_input)[0]

    st.write('### Random Forest Model')
    if rf_prediction == 1:
        st.success(f'Prediction: \u2705 Survived (Probability: {rf_survived_proba:.2f})')
    else:
        st.error(f'Prediction: \u274C Not Survived (Probability: {1-rf_survived_proba:.2f})')

    st.subheader('Raw Processed Input (for debugging)')
    st.write(processed_input)