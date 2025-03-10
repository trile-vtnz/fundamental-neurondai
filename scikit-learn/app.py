import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset (using Iris dataset as an example)
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# Streamlit UI
st.title("🌸 Iris Species Prediction App")
st.write("Explore the famous Iris dataset and predict species interactively!")

# Show dataset preview with interactive checkbox
if st.checkbox("Show Raw Dataset"):
    st.write(df.head())

# Data preprocessing with interactive learning
st.subheader("🔍 Data Preprocessing")

# Standardizing the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_scaled['target'] = df['target']

if st.checkbox("Show Scaled Data"):
    st.write(df_scaled.head())

# Visualization - Pairplot
st.subheader("📊 Data Visualization")
if st.checkbox("Show Pairplot"):
    fig = sns.pairplot(df, hue='target', diag_kind='kde', palette='husl')
    st.pyplot(fig)

# Interactive model parameters selection
test_size = st.slider("Test Size (Proportion of data for testing)", 0.1, 0.5, 0.2, 0.05)
random_state = st.number_input("Random State (for reproducibility)", min_value=0, max_value=100, value=42)

if st.button("Train Model"):
    # Splitting the dataset
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)

    # Store in session state
    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.is_trained = True
    st.session_state.y_pred = y_pred

if st.session_state.is_trained:
    accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
    st.write(f"✅ **Model Accuracy:** {accuracy:.2f}")
    st.text(classification_report(st.session_state.y_test, st.session_state.y_pred))

    # Feature importance visualization
    st.subheader("🔬 Feature Importance")
    feature_importances = st.session_state.model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(df.columns[:-1], feature_importances, color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance in RandomForest Model")
    st.pyplot(fig)

if st.button("Show model param"):
    st.write(st.session_state.model.get_params())

# User input for prediction
st.subheader("🎯 Make a Prediction")
sepal_length = st.slider("Sepal Length", float(df.iloc[:, 0].min()), float(df.iloc[:, 0].max()), float(df.iloc[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(df.iloc[:, 1].min()), float(df.iloc[:, 1].max()), float(df.iloc[:, 1].mean()))
petal_length = st.slider("Petal Length", float(df.iloc[:, 2].min()), float(df.iloc[:, 2].max()), float(df.iloc[:, 2].mean()))
petal_width = st.slider("Petal Width", float(df.iloc[:, 3].min()), float(df.iloc[:, 3].max()), float(df.iloc[:, 3].mean()))

if st.session_state.is_trained:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = st.session_state.model.predict(input_data)
    predicted_species = data.target_names[prediction[0]]
    st.write(f"🌼 **Predicted Species:** {predicted_species}")
else:
    st.warning("⚠️ Please train the model first before making a prediction!")