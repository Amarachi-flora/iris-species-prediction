# Step 1: Load the dataset
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Step 2: Split the data
from sklearn.model_selection import train_test_split

X = iris.data  # Features (sepal/petal measurements)
y = iris.target  # Target (species)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Choose and train a model using Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:")
print(cm)

# Step 6: Input interface (Console version)
print("\n --- Predict Your Own Iris Flower ---")
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
species = iris.target_names[prediction[0]]
print(f"\n Predicted species: {species}")
