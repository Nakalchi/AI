from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

# Train model if it doesn't exist
if not os.path.exists("model.joblib"):
    print("Training model...")

    X, y = load_iris(return_X_y=True)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "model.joblib")

    print("Model saved successfully.")

# Load model
model = joblib.load("model.joblib")

# Sample input
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample)

flower_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

print("\n===== Prediction Result =====")
print("Input:", sample.tolist())
print("Predicted Class:", prediction[0])
print("Flower Name:", flower_names[prediction[0]])
print("=============================")
