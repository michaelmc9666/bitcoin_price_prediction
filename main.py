import streamlit as st
import numpy as np
import zipfile
import kagglehub
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_wine
from sklearn.utils import Bunch
from alibi.explainers.anchor_tabular import AnchorTabular
import pandas as pd

def csv_to_bunch(csv_file):
    # read csv into pandas dataframe
    pd_data = pd.read_csv(csv_file)

    # finds any invalid data, and converts them to the mean of that column
    pd_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    pd_data.fillna(pd_data.mean(), inplace=True)

    feature_vars = pd_data.iloc[:, [0, 1, 2, 3, 5]].values        # all columns except the close price as features
    target_var   = pd_data.iloc[:, 4].values                 # the close_price column is set as the target variable

    # feature names and target names defined
    feature_names = list(pd_data.columns[:-1])
    target_names = 'Closing price'

    return Bunch(
        data = feature_vars,
        target = target_var,
        feature_names = feature_names,
        target_names = target_names,
        DESCR="bitcoin-data"
    )

# provide csv file path
csv_file_path = 'btcusd_1-min_data.csv'

# Load the wine dataset and split into train/test sets
data = csv_to_bunch(csv_file_path)

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Set up the AnchorTabular explainer with predictor and feature names
explainer = AnchorTabular(predictor=model.predict, feature_names=data.feature_names)
explainer.fit(X_train, disc_perc=(10, 20, 30, 40, 50, 60, 70, 80, 90))  # Fit explainer to training data

# Streamlit app UI setup
st.title("Anchor Explanations for Bitcoin Dataset")
st.write("Use this app to generate anchor explanations for instances in the Wine dataset.")

# Select an instance to explain
instance_index = st.slider("Select instance index", 0, len(X_test) - 1, 0)
instance = X_test[instance_index].reshape(1, -1)

# Display the selected instance
st.write("Instance to explain:")
st.write({data.feature_names[i]: instance[0][i] for i in range(len(data.feature_names))})

# Display the predicted class for the selected instance
predicted_value = format(model.predict(instance)[0], ".2f")
st.write(f"Predicted Value: ${predicted_value}")  # Display wine type

# Generate and display the explanation
if st.button("Generate Explanation"):
    explanation = explainer.explain(instance)
    st.write("Anchor Explanation:", explanation.anchor)
    st.write("Precision:", explanation.precision)
    st.write("Coverage:", explanation.coverage)
