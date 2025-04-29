# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the datasets
train = pd.read_csv('D:/Projects/Credit_Card_Fraud_Detection/data/fraudTrain.csv')
test = pd.read_csv('D:/Projects/Credit_Card_Fraud_Detection/data/fraudTest.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Preprocessing
train.rename(columns={'trans_date_trans_time': 'transaction_time'}, inplace=True)
test.rename(columns={'trans_date_trans_time': 'transaction_time'}, inplace=True)

train['transaction_time'] = pd.to_datetime(train['transaction_time'], errors='coerce')
test['transaction_time'] = pd.to_datetime(test['transaction_time'], errors='coerce')

# Feature Engineering
train['hour'] = train['transaction_time'].dt.hour
train['day'] = train['transaction_time'].dt.day
train['month'] = train['transaction_time'].dt.month

test['hour'] = test['transaction_time'].dt.hour
test['day'] = test['transaction_time'].dt.day
test['month'] = test['transaction_time'].dt.month

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'first', 'last', 'gender', 'street', 'city', 'state', 
             'zip', 'job', 'dob', 'trans_num', 'transaction_time']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# Label Encoding for high-cardinality categorical features
le_merchant = LabelEncoder()
le_category = LabelEncoder()

train['merchant'] = le_merchant.fit_transform(train['merchant'])
test['merchant'] = le_merchant.transform(test['merchant'])

train['category'] = le_category.fit_transform(train['category'])
test['category'] = le_category.transform(test['category'])

# Align train and test
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Drop rows with missing target
train = train.dropna(subset=['is_fraud'])

# Prepare Features and Labels
X = train.drop('is_fraud', axis=1)
y = train['is_fraud']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test = test.drop('is_fraud', axis=1)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE carefully by limiting sample size to avoid memory error
print("\nApplying SMOTE... (with reduced sample)")
smote = SMOTE(random_state=42, sampling_strategy=0.1)  # Only balance to 10% to avoid memory issue
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Shape after SMOTE: {X_train_resampled.shape}")

# Define Random Forest and RandomizedSearch parameters
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

random_search.fit(X_train_resampled, y_train_resampled)

print("\nBest parameters found:", random_search.best_params_)

# Best model
best_rf = random_search.best_estimator_

# Validation
y_val_pred = best_rf.predict(X_val_scaled)

print("\n--- Validation Performance ---")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("ROC AUC:", roc_auc_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Now predicting on test data
# (Test does not have 'is_fraud' label, so we predict and can only save or use later)
y_test_pred = best_rf.predict(X_test_scaled)

# Save model
joblib.dump(best_rf, 'fraud_detection_model.pkl')
print("\nModel saved as 'fraud_detection_model.pkl'.")

# Save the scaler too (important for future inference)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'.")

# Predict function
def predict_fraud(new_data):
    # Load scaler and model if needed
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('fraud_detection_model.pkl')
    
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)

# Example (Commented out, for future use)
# new_data = pd.DataFrame([...])
# print(predict_fraud(new_data))
