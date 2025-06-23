import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

# Load dataset
df = pd.read_csv('Austim Dataset.csv')

# Drop unnecessary columns
df = df.drop(columns=['ID', 'age_desc'])

# Fix country names
mapping = {
    'Viet Nam': 'Vietnam',
    'AmericanSamao': 'United States',
    'Hong Kong': 'China'
}
df['contry_of_res'] = df['contry_of_res'].replace(mapping)

# Fill missing values with mode for categorical columns
def fill_mode(df, col):
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)
    return df
for col in ['ethnicity', 'relation']:
    df = fill_mode(df, col)

# Encode categorical features
categorical_cols = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop(columns=['Class/ASD'])
y = df['Class/ASD']

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train Random Forest (as in notebook)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_res, y_res)

# Save model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print('Model and encoders exported successfully.') 