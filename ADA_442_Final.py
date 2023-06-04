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
import io

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
df['default'] = df['default'].map({'yes': 1, 'no': 0})
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Display the updated dataframe
st.write(df.head())

# Dump unnnecessery rows
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'contact', 'month'])

# Display the updated dataframe
st.write(df.head())

# Check for duplicate rows
duplicates = df.duplicated().sum()

# Display the number of duplicate rows
st.write(f"Number of duplicate rows: {duplicates}")

# Get dataframe info
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

# Display the dataframe info
st.text(s)

# Generate the correlation matrix
correlation_matrix = df.corr()

# Select specific rows and columns to display
columns_to_display = ['age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'previous', 'y']
correlation_matrix = correlation_matrix.loc[columns_to_display, columns_to_display]

# Display the correlation matrix in the Streamlit app
st.dataframe(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)

# Display the plot in Streamlit
st.pyplot(plt)

# Remove one of two features that have a correlation higher than 0.9
columns = np.full((correlation_matrix.shape[0],), True, dtype=bool)
for i in range(correlation_matrix.shape[0]):
    for j in range(i+1, correlation_matrix.shape[0]):
        if correlation_matrix.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df_clean.columns[columns]
df_clean = df_clean[selected_columns]

# Display the updated dataframe
st.dataframe(df_clean)


# Split the dataframe into inputs (X) and output (y)
X = df_clean.drop('y', axis=1)
y = df_clean['y']

# Train a RandomForestClassifier to get feature importances
clf = RandomForestClassifier()
clf.fit(X, y)
importances = clf.feature_importances_

# Create a series with the feature importances
f_importances = pd.Series(importances, X.columns)

# Sort the series by importance
f_importances.sort_values(ascending=False, inplace=True)

# Plot the feature importances
plt.figure(figsize=(16,9))
f_importances.plot(kind='bar')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Neural Network": MLPClassifier()
}

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through the models
for model_name, model in models.items():
    st.write(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    st.write('Accuracy:', accuracy_score(y_test, y_pred))
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))
    st.write(f'{model_name} RMSE: {rmse:.2f}')
    
    
    
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample dataset with SMOTE to improve performance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Use best parameters For our Dataset
rfc = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', random_state=42)

rfc.fit(X_resampled, y_resampled)

y_pred = rfc.predict(X_test)
auc_roc = metrics.roc_auc_score(y_test, y_pred)

# Display the results in Streamlit
st.write("Accuracy: ", accuracy_score(y_test, y_pred))
st.write("Precision:", metrics.precision_score(y_test, y_pred))
st.write("Recall:", metrics.recall_score(y_test, y_pred))
st.write("F1 Score:", metrics.f1_score(y_test, y_pred))
st.write("AUC-ROC:", auc_roc)
st.write("Confusion Matrix:")
st.table(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))


