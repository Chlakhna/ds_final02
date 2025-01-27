
import streamlit as st
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Load and cache the model
@st.cache_resource
def load_model():
    model = LogisticRegressionModel(input_dim=7, num_classes=7)
    model.load_state_dict(torch.load("C:/Users/MSI/Downloads/logistic_regression_model.pth"))
    model.eval()  # Set the model to evaluation mode
    return model

# Load and cache the dataset
@st.cache_data
def load_dataset():
    df = pd.read_excel("C:/Users/MSI/Downloads/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx")
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y

# Load and cache the scaler and PCA
@st.cache_data
def get_scaler_and_pca(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    normalized_X = scaler.transform(X)
    pca_ = PCA(n_components=7)
    pca_.fit(normalized_X)
    return scaler, pca_

# Main script
st.title("🌱 Dry Bean Class Prediction")
st.write("Enter the values for each feature to predict the class of the dry bean.")

# Load cached resources
model = load_model()
X, y = load_dataset()
scaler, pca_ = get_scaler_and_pca(X)

# Collect user inputs
area = st.slider("Area", min_value=0.0, max_value=500.0, step=0.1)
perimeter = st.slider("Perimeter", min_value=0.0, max_value=500.0, step=0.1)
major_axis_length = st.slider("Major Axis Length", min_value=0.0, max_value=500.0, step=0.1)
minor_axis_length = st.slider("Minor Axis Length", min_value=0.0, max_value=500.0, step=0.1)
aspect_ratio = st.slider("Aspect Ratio", min_value=0.0, max_value=10.0, step=0.01)
eccentricity = st.slider("Eccentricity", min_value=0.0, max_value=1.0, step=0.01)
convex_area = st.slider("Convex Area", min_value=0.0, max_value=500.0, step=0.1)
equiv_diameter = st.slider("Equiv Diameter", min_value=0.0, max_value=100.0, step=0.1)
extent = st.slider("Extent", min_value=0.0, max_value=1.0, step=0.01)
solidity = st.slider("Solidity", min_value=0.0, max_value=1.0, step=0.01)
roundness = st.slider("Roundness", min_value=0.0, max_value=1.0, step=0.01)
compactness = st.slider("Compactness", min_value=0.0, max_value=1.0, step=0.01)
shape_factor_1 = st.slider("Shape Factor 1", min_value=0.0, max_value=1.0, step=0.01)
shape_factor_2 = st.slider("Shape Factor 2", min_value=0.0, max_value=1.0, step=0.01)
shape_factor_3 = st.slider("Shape Factor 3", min_value=0.0, max_value=1.0, step=0.01)
shape_factor_4 = st.slider("Shape Factor 4", min_value=0.0, max_value=1.0, step=0.01)

# Create a DataFrame from the user input
input_data = pd.DataFrame({
    'Area': [area],
    'Perimeter': [perimeter],
    'MajorAxisLength': [major_axis_length],
    'MinorAxisLength': [minor_axis_length],
    'AspectRation': [aspect_ratio],
    'Eccentricity': [eccentricity],
    'ConvexArea': [convex_area],
    'EquivDiameter': [equiv_diameter],
    'Extent': [extent],
    'Solidity': [solidity],
    'roundness': [roundness],
    'Compactness': [compactness],
    'ShapeFactor1': [shape_factor_1],
    'ShapeFactor2': [shape_factor_2],
    'ShapeFactor3': [shape_factor_3],
    'ShapeFactor4': [shape_factor_4]
})

# Button to trigger prediction
if st.button("🔮 Predict Class", key="predict_button"):
    # Normalize and apply PCA to the input data
    normalized_input = scaler.transform(input_data)
    input_pca = pca_.transform(normalized_input)
    input_tensor = torch.tensor(input_pca, dtype=torch.float32)

    # Predict the class
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    # Map the class index to the class name
    class_map = {0: 'SIRA', 1: 'BOMBAY', 2: 'DERMASON', 3: 'BARBUNYA', 4: 'HOROZ', 5: 'CALI', 6: 'SEKER'}
    predicted_class_label = class_map[predicted_class.item()]

    # Display the predicted class
    st.success(f"🌟 Predicted Class: **{predicted_class_label}**")
