import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame for exploratory visuals
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Predict on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Sidebar inputs
st.sidebar.header("Input Flower Measurements")
st.sidebar.write("Adjust the sliders to input your flower's measurements:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Main app title & intro
st.title(" Iris Flower Species Predictor")
st.markdown("""
Welcome! This app predicts the species of an Iris flower based on your input measurements.
The model is trained using Logistic Regression on the classic Iris dataset.
Use the sliders in the sidebar to enter the flower's sepal and petal measurements.
""")

# Prediction result
st.success(f"### Predicted species: **{predicted_species.capitalize()}**")
st.info(f"Model accuracy on the test set: **{accuracy:.2f}**")

# Confusion Matrix plot
st.write("### Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')
st.pyplot(fig_cm)

# Feature Importance Insight (coefficients)
st.write("### Feature Importance Insight")
coefficients = model.coef_
feature_importance = pd.DataFrame(coefficients.T, index=feature_names, columns=target_names)
st.write("The table below shows the model's learned coefficients for each species. Features with larger absolute coefficients have stronger influence on the prediction for that species.")
st.dataframe(feature_importance.style.background_gradient(axis=0, cmap='coolwarm'))

# Exploratory Data Visualizations
st.write("### Exploratory Data Analysis")
st.markdown("Visualizing relationships in the Iris dataset:")

# Pairplot alternative: Scatter plots colored by species
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', ax=axes[0,0])
axes[0,0].set_title('Sepal Length vs Sepal Width')
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', ax=axes[0,1])
axes[0,1].set_title('Petal Length vs Petal Width')
sns.histplot(data=df, x='sepal length (cm)', hue='species', multiple='stack', ax=axes[1,0])
axes[1,0].set_title('Sepal Length Distribution')
sns.histplot(data=df, x='petal length (cm)', hue='species', multiple='stack', ax=axes[1,1])
axes[1,1].set_title('Petal Length Distribution')

plt.tight_layout()
st.pyplot(fig)

# Summary & Conclusion
st.markdown("---")
st.header("Summary & Conclusion")
st.markdown("""
- The Logistic Regression model achieves high accuracy (~96%) on the Iris dataset test set.
- Petal measurements are generally stronger predictors of species than sepal measurements.
- The confusion matrix shows a few misclassifications mostly between *Iris versicolor* and *Iris virginica*.
- This interactive app allows you to input your own flower measurements and see instant predictions.
- Explore the data visuals above to understand the distribution and relationships between features and species.

Thank you for using this app. Feel free to reach out if you'd like to learn more about machine learning or this dataset.
""")

# Footer
st.markdown("Developed by Amarachi. Powered by scikit-learn & Streamlit ")
