import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title("Streamlit Demo")

# Sidebar for additional inputs
st.sidebar.header("Sidebar Controls")
theme_choice = st.sidebar.selectbox("Choose a Theme", ["Light", "Dark", "Custom"])
st.sidebar.write(f"Selected Theme: {theme_choice}")

# Main content area
st.header("Common Widgets in Streamlit")

# Text Input
name = st.text_input("Enter your name:", placeholder= "Type here...")
if name != "Type here...":
    st.write(f"Hello, {name}!")

# Slider
age = st.slider("Select your age:", 0, 100, 25)
st.write(f"Your age is: {age}")

# Checkbox
if st.checkbox("Show some random data"):
    # Create a small dataframe with random data
    df = pd.DataFrame(
        np.random.randn(5, 3),
        columns=["A", "B", "C"]
    )
    st.dataframe(df)

# Selectbox
department = st.selectbox("Choose your organization:", ["OrientSoftware", "NeurondAI"])
st.write(f"Your department is : {department}")

# Radio Buttons
mood = st.radio("How are you feeling today?", ["Happy 😊", "Neutral 😐", "Sad 😢"])
st.write(f"You're feeling: {mood}")

# Button
if st.button("Click me!"):
    st.write("Welcome to Orient Software Development Corporation 😊")

# File Uploader
uploaded_file = st.file_uploader("Upload a file (optional):", type=["txt", "csv"])
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    if uploaded_file.type == "text/csv":
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
    st.success("Thanks for exploring Streamlit widgets!")
    st.balloons()

# Add a footer
st.markdown("---")
st.write("Tri Duc Le - NeurondAI")