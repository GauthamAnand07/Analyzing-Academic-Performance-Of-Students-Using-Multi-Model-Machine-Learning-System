import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

    
# Set page configuration
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="📚",
    layout="centered",  # wide
    initial_sidebar_state="expanded")  # auto

# Load the dataset
df = pd.read_csv("StudentsPerformanceModified.csv")

# Check if all required columns exist
required_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 
                    'writing score', 'reading score', 'project score', 'ecc score', 'intern score', 'attendance']
missing_columns = [col for col in required_columns if col not in df.columns]

# List of extra curricular activities
activities = ['Cricket', 'Football', 'Athletics', 'Hockey']

# Create a DataFrame
df_activities = pd.DataFrame(activities, columns=['extra curricular activities'])

if missing_columns:
    st.error(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
else:
    # Define criteria for performance classification
    def classify_performance(score):
        if score >= 90:
            return 'Very Good'
        elif score >= 75:
            return 'Good'
        elif score >= 50:
            return 'Average'
        else:
            return 'Bad'
    

    # Split features and target variable
    X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 
            'writing score', 'reading score', 'project score', 'ecc score', 'intern score', 'attendance']]
    y = df['writing score'].apply(classify_performance)  # Classify writing score into categories

    # Encode categorical variables
    label_encoders = {}
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Define the Streamlit app
    st.title("Academic Performance Predictor")
    st.write("Enter the details of the student")

    # Streamlit input widgets
    gender = st.selectbox('Gender:', df['gender'].unique())
    race_ethnicity = st.selectbox('Race/Ethnicity:', df['race/ethnicity'].unique())
    parent_education = st.selectbox('Parental Education Level:', df['parental level of education'].unique())
    lunch = st.selectbox('Type of Lunch:', df['lunch'].unique())
    test_prep = st.selectbox('Test Preparation Course:', df['test preparation course'].unique())
    writing_score = st.slider('Writing Score:', min_value=0, max_value=100, step=1)
    reading_score = st.slider('Reading Score:', min_value=0, max_value=100, step=1)
    project_score = st.slider('Project Score:', min_value=0, max_value=100, step=1)
    extra_curricular_activities = st.multiselect('Extra Curricular Activities:', df_activities['extra curricular activities'].unique())
    ecc_score = st.slider('ECC Score:', min_value=0, max_value=100, step=1)
    intern_score = st.slider('Intern Score:', min_value=0, max_value=100, step=1)
    attendance = st.slider('Attendance:', min_value=0, max_value=100, step=1)


    # Prepare input features for prediction
    input_features = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race_ethnicity],
        'parental level of education': [parent_education],
        'lunch': [lunch],
        'test preparation course': [test_prep],
        'writing score': [writing_score],
        'reading score': [reading_score],
        'project score': [project_score],
        'ecc score': [ecc_score],
        'intern score': [intern_score],
        'attendance': [attendance]
    })

    # Encode input features using label encoders
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        input_features[col] = label_encoders[col].transform(input_features[col])

    # Make predictions
    if st.button('Predict'):
        prediction = rf.predict(input_features)
        st.write(f"The predicted performance category is: {prediction[0]}")
