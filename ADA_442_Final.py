#!/usr/bin/env python
# coding: utf-8

# In[22]:

from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

import streamlit as st





# In[23]:


df = pd.read_csv('bank-full.csv', delimiter=';', quotechar='"')


# In[24]:


df.head() ;
df.info();
df.describe()


# In[4]:


df.isnull().sum() 
##null value check


# In[25]:



##replace unknown values to get better results
df['contact'].replace('unknown', df['contact'].mode()[0], inplace=True)
df['poutcome'].replace('unknown', df['poutcome'].mode()[0], inplace=True)

df.head()


# In[26]:



##unneccesery field with lots of unknown values

df=df.drop(['poutcome'], axis=1)
df.head()


# In[27]:


## Implement Z-score for Handle outliers
from scipy import stats
import numpy as np

# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Data Preprocessing")

# Load the dataset
df = ...  # Add appropriate code here to load the dataset

# Drop unnecessary field with lots of unknown values
df = df.drop(['poutcome'], axis=1)

# Implement Z-score for handling outliers
numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
df_numeric = df[numeric_cols]

z_scores = np.abs(stats.zscore(df_numeric))

df_clean = df[(z_scores < 3).all(axis=1)]

# Encode yes-no rows to 0-1
df_clean['default'] = df_clean['default'].map({'yes': 1, 'no': 0})
df_clean['housing'] = df_clean['housing'].map({'yes': 1, 'no': 0})
df_clean['loan'] = df_clean['loan'].map({'yes': 1, 'no': 0})
df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0})

# Drop unnecessary rows
df_clean = pd.get_dummies(df_clean, columns=['job', 'marital', 'education', 'contact', 'month'])

# Check for duplicate values
duplicate_count = df_clean.duplicated().sum()

# Display information about the cleaned dataset
st.write("Cleaned Dataset Information:")
st.write(df_clean.info())

# To get better correlation matrix values
correlation_matrix = df_clean.corr()
columns_to_display = ['age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'previous', 'y']
rows_to_display = ['age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'previous', 'y']

# Display correlation matrix
st.write("Correlation Matrix:")
st.write(correlation_matrix.loc[rows_to_display, columns_to_display])


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Correlation Matrix")

# Load the dataset
correlation_matrix = ... # Add appropriate code here to load the correlation matrix
rows_to_display = ...    # Add appropriate code here to determine the rows to display
columns_to_display = ... # Add appropriate code here to determine the columns to display

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(12, 10))

# Filter the correlation matrix
filtered_matrix = correlation_matrix.loc[rows_to_display, columns_to_display]

# Create a Seaborn heatmap
sns.heatmap(filtered_matrix, annot=True, cmap=plt.cm.CMRmap_r, ax=ax)

# Send the figure to the Streamlit interface
st.pyplot(fig)


# In[34]:


## Remove one of two features that have a correlation higher than 0.9 or -0.9

# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Column Selection")

# Load the dataset
df_clean = ...  # Add appropriate code here to load the dataset
correlation_matrix = ...  # Add appropriate code here to load the correlation matrix

# Column selection logic
columns = np.full((correlation_matrix.shape[0],), True, dtype=bool)
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[0]):
        if correlation_matrix.iloc[i, j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df_clean.columns[columns]
df_clean = df_clean[selected_columns]

# Display the selected columns
st.write("Selected Columns:")
st.write(df_clean)



# In[36]:


## Implement Feature Importance using Tree-Based Classifiers for get view of more informative to less informative columns
from sklearn.ensemble import RandomForestClassifier


# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Feature Importance")

# Load the dataset
df_clean = ...  # Add appropriate code here to load the dataset

# Separate features and target variable
X = df_clean.drop('y', axis=1)
y = df_clean['y']

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Get feature importances
importances = clf.feature_importances_

# Create a pandas Series for feature importances
f_importances = pd.Series(importances, X.columns)

# Sort feature importances in descending order
f_importances.sort_values(ascending=False, inplace=True)

# Create a bar plot of feature importances
fig, ax = plt.subplots(figsize=(16, 9))
f_importances.plot(x='Features', y='Importance', kind='bar', ax=ax, rot=45)
plt.tight_layout()

# Send the figure to the Streamlit interface
st.pyplot(fig)

# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error

# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Model Search")

# Load the dataset
df_clean = ...  # Add appropriate code here to load the dataset

# Separate features and target variable
X = df_clean.drop('y', axis=1)
y = df_clean['y']

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
    st.write(classification_report(y_test, y_pred))
    st.write(f'{model_name} RMSE: {rmse:.2f}')
    st.write("\n")


# In[42]:



## Implement Random Forest and find best parameters with GridSearch
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV

# Define the parameter grid
#param_grid = {
    #   'n_estimators': [100, 200, 300, 400, 500],
    #'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth': [10, 20, 30, 40, 50, None],
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4]
#}


#rf = RandomForestClassifier(random_state=42)

#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                          #cv=3, n_jobs=-1, verbose=2)
#grid_search.fit(X_train, y_train)

#best_params = grid_search.best_params_

#print("Best parameters found: ", best_params)

#best_rf_model = RandomForestClassifier(**best_params)

#best_rf_model.fit(X_train, y_train)

#y_pred = best_rf_model.predict(X_test)


# In[50]:
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics

# Set up Streamlit interface
st.set_page_config(layout="wide")
st.title("Best Parameters")

# Load the dataset
df_clean = ...  # Add appropriate code here to load the dataset

# Separate features and target variable
X = df_clean.drop('y', axis=1)
y = df_clean['y']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample dataset with SMOTE to improve performance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Create a random forest classifier with best parameters
rfc = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', random_state=42)

rfc.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = rfc.predict(X_test)
auc_roc = metrics.roc_auc_score(y_test, y_pred)

# Display evaluation metrics
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", metrics.precision_score(y_test, y_pred))
st.write("Recall:", metrics.recall_score(y_test, y_pred))
st.write("F1 Score:", metrics.f1_score(y_test, y_pred))
st.write("AUC-ROC:", auc_roc)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))
# In[ ]:




