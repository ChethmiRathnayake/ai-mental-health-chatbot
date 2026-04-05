import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib  
import pandas as pd

# Load the trained model
model = joblib.load('HearRate_model.pkl')

# Input
new_data = pd.DataFrame({
    'Age': [25],
    'Gender': ['Female'],
    'Height': [165],
    'Weight': [70],
    'BMI': [25.7],
    'BloodType': ['O+'],
    'HeartRate': [72]
})

# Predict activities using the trained model
predicted_activities = model.predict(new_data)
print(f'Recommended Activities: {predicted_activities[0]}')
