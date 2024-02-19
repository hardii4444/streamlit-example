import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit App
st.title('Iris Flower Classification')
st.sidebar.header('User Input')

# Define user input components
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Make predictions
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display results
st.subheader('Predicted Class')
st.write(iris.target_names[prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)
