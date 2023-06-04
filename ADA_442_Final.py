import streamlit as st
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('bank-full.csv', delimiter=';', quotechar='"')
    return df

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
    for model_name, model in models.items():
        st.write(f"Training {model_name}...")
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write(classification_report(y_test, y_pred))
        st.write(f'{model_name} RMSE: {rmse:.2f}\n')

def main():
    st.title("Bank Marketing Dataset Analysis")
    df = load_data()
    st.write("Data Descriptive Statistics", df.describe())
    df_clean = clean_data(df)
    st.write("Cleaned First 5 Records", df_clean.head())
    fig = plot_importances(df_clean)
    st.pyplot(fig.figure)
    train_and_predict(df_clean)

if __name__ == '__main__':
    main()
