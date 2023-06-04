#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_squared_error


@st.cache
def load_data():
    df = pd.read_csv('bank-full.csv', delimiter=';', quotechar='"')
    return df

@st.cache
def clean_data(df):
    df['contact'].replace('unknown', df['contact'].mode()[0], inplace=True)
    df['poutcome'].replace('unknown', df['poutcome'].mode()[0], inplace=True)
    df=df.drop(['poutcome'], axis=1)
    numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    df_numeric = df[numeric_cols]
    z_scores = np.abs(stats.zscore(df_numeric))
    df_clean = df[(z_scores < 3).all(axis=1)]
    df_clean['default'] = df_clean['default'].map({'yes': 1, 'no': 0})
    df_clean['housing'] = df_clean['housing'].map({'yes': 1, 'no': 0})
    df_clean['loan'] = df_clean['loan'].map({'yes': 1, 'no': 0})
    df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0})
    df_clean = pd.get_dummies(df_clean, columns=['job', 'marital', 'education', 'contact', 'month'])
    correlation_matrix = df_clean.corr()
    columns = np.full((correlation_matrix.shape[0],), True, dtype=bool)
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[0]):
            if correlation_matrix.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df_clean.columns[columns]
    df_clean = df_clean[selected_columns]
    return df_clean

def plot_importances(df_clean):
    X = df_clean.drop('y', axis=1)
    y = df_clean['y']
    clf = RandomForestClassifier()
    clf.fit(X, y)
    importances = clf.feature_importances_
    f_importances = pd.Series(importances, X.columns)
    f_importances.sort_values(ascending=False, inplace=True)
    fig = f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=45)
    return fig

def train_and_predict(df_clean):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "Neural Network": MLPClassifier()
    }
    X = df_clean.drop('y', axis=1)
    y = df_clean['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    rfc = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=42)
    rfc.fit(X_resampled, y_resampled)
    y_pred = rfc.predict(X_test)
    auc_roc = metrics.roc_auc_score(y_test, y_pred)
    metrics_data = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": metrics.precision_score(y_test, y_pred),
        "Recall": metrics.recall_score(y_test, y_pred),
        "F1 Score": metrics.f1_score(y_test, y_pred),
        "AUC-ROC": auc_roc,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred)
    }
    return metrics_data

def main():
    st.title("Bank Marketing Data Analysis")
    df = load_data()
    df_clean = clean_data(df)
    st.header("Feature Importances")
    plot_importances(df_clean)
    st.header("Model Training and Prediction Metrics")
    metrics_data = train_and_predict(df_clean)
    for key, value in metrics_data.items():
        st.subheader(key)
        st.write(value)

if __name__ == "__main__":
    main()
