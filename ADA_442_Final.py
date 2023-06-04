#!/usr/bin/env python
# coding: utf-8

# In[22]:
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Bank Marketing Data Analysis")

# Load the dataset
df = pd.read_csv('bank-full.csv', delimiter=';', quotechar='"')

# Display the dataset
st.write("Original Dataset:")
st.write(df)

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")
st.write("Summary Statistics:")
st.write(df.describe())

# Check for missing values
st.write("Missing Values:")
st.write(df.isnull().sum())

# Data Preprocessing
st.subheader("Data Preprocessing")

# Replace unknown values with mode
df['contact'].replace('unknown', df['contact'].mode()[0], inplace=True)
df['poutcome'].replace('unknown', df['poutcome'].mode()[0], inplace=True)

# Drop unnecessary column with lots of unknown values
df = df.drop('poutcome', axis=1)

# Implement Z-score for handling outliers
numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split the dataset into features and target variable
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = rf_classifier.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
classification_report = classification_report(y, y_pred)
confusion_mat = confusion_matrix(y, y_pred)

# Display the evaluation results
st.subheader("Model Evaluation")
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_report)
st.write("Confusion Matrix:")
st.write(confusion_mat)

# Feature Importance
st.subheader("Feature Importance")

# Get feature importances from the random forest classifier
importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create a pandas DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by feature importance
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Create a bar plot of feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

# Display the feature importances plot
st.pyplot(plt)



