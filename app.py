import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------
# Load Models & Constants
# ------------------------------------------------------
best_logreg = joblib.load('best_logreg_model.pkl')
best_rf = joblib.load('best_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

AGE_MEDIAN = 28.0
FARE_MEDIAN = 14.4542
EMBARKED_MODE = 'S'
FARE_BINS = [0.0, 7.9104, 14.4542, 31.0, 512.3292]
FARE_LABELS = ['Low', 'Medium', 'High', 'Very High']

title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Rare', 'Sir': 'Rare',
    'Mme': 'Mrs', 'Don': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare'
}

# ------------------------------------------------------
# Processing Function
# ------------------------------------------------------
def process_user_input(input_data):
    df_user = pd.DataFrame([input_data])

    df_user['Name'] = df_user['Name'].fillna('Mr. User')
    df_user['Cabin'] = df_user['Cabin'].replace('', np.nan)

    # Title extraction
    df_user['Title'] = df_user['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df_user['Title'] = df_user['Title'].map(title_mapping).fillna('Rare')

    # Family
    df_user['FamilySize'] = df_user['SibSp'] + df_user['Parch'] + 1
    df_user['IsAlone'] = (df_user['FamilySize'] == 1).astype(int)

    # Cabin features
    df_user['Deck'] = df_user['Cabin'].astype(str).str[0]
    df_user['Deck'] = df_user['Deck'].fillna('Unknown')
    df_user['HasCabin'] = df_user['Cabin'].notna().astype(int)

    # Age / Fare groups
    df_user['Age'] = df_user['Age'].fillna(AGE_MEDIAN)
    df_user['AgeGroup'] = pd.cut(
        df_user['Age'], 
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior']
    )

    df_user['Fare'] = df_user['Fare'].fillna(FARE_MEDIAN)
    df_user['FareGroup'] = pd.cut(
        df_user['Fare'],
        bins=FARE_BINS,
        labels=FARE_LABELS,
        include_lowest=True
    )

    cols_to_keep = ['Pclass', 'Sex', 'Embarked', 'Title',
                    'FamilySize', 'IsAlone', 'HasCabin',
                    'AgeGroup', 'FareGroup']
    
    df_processed = df_user[cols_to_keep]

    # One-hot encode
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

    # Align with training columns
    missing_cols = set(model_columns) - set(df_encoded.columns)
    for c in missing_cols:
        df_encoded[c] = 0

    extra_cols = set(df_encoded.columns) - set(model_columns)
    df_encoded = df_encoded.drop(columns=list(extra_cols))

    df_final = df_encoded[model_columns]

    return df_final

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
)

# Header
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #003366;
}
.sub-text {
    text-align: center;
    font-size: 18px;
    color: #444;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f8f9fa;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Enter passenger details to estimate the chance of surviving the Titanic disaster.</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🧍 Passenger Details")
st.sidebar.write("---")

pclass = st.sidebar.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.sidebar.radio('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0, 80, 25)
sibsp = st.sidebar.slider('Siblings/Spouses Aboard', 0, 8, 0)
parch = st.sidebar.slider('Parents/Children Aboard', 0, 6, 0)
fare = st.sidebar.number_input('Fare Paid', min_value=0.0, max_value=500.0, value=30.0)
cabin = st.sidebar.text_input('Cabin (optional)', '')
embarked = st.sidebar.selectbox('Embarked From', ['S', 'C', 'Q'])

# Input dict
user_input = {
    'PassengerId': 0,
    'Pclass': pclass,
    'Name': 'Mr. User' if sex == 'male' else 'Mrs. User',
    'Sex': sex,
    'Age': float(age),
    'SibSp': sibsp,
    'Parch': parch,
    'Ticket': '0',
    'Fare': float(fare),
    'Cabin': cabin,
    'Embarked': embarked
}

st.write("")

# ------------------------------------------------------
# Predictions
# ------------------------------------------------------
if st.sidebar.button("🔮 Predict Survival"):
    processed = process_user_input(user_input)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Results")

    # Logistic Regression
    log_proba = best_logreg.predict_proba(processed)[0][1]
    log_pred = best_logreg.predict(processed)[0]

    # Random Forest
    rf_proba = best_rf.predict_proba(processed)[0][1]
    rf_pred = best_rf.predict(processed)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚖️ Logistic Regression")
        st.metric(
            label="Survival Probability",
            value=f"{log_proba*100:.1f}%",
            delta="Survived" if log_pred==1 else "Not Survived"
        )

    with col2:
        st.markdown("### 🌲 Random Forest")
        st.metric(
            label="Survival Probability",
            value=f"{rf_proba*100:.1f}%",
            delta="Survived" if rf_pred==1 else "Not Survived"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔧 Show Processed Model Input"):
        st.write(processed)

else:
    st.info("Click **Predict Survival** from the sidebar to get results.")
