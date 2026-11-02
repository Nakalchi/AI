#pip install scikit-learn pandas fastapi uvicorn joblib numpy
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, "iris_model.pkl")
print("Model saved as iris_model.pkl")
model = joblib.load("iris_model.pkl")
app = FastAPI()
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
@app.post("/predict/")
def predict(iris: IrisRequest):
    features = np.array([
        [
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width
        ]
    ])
    prediction = model.predict(features)
    species = [
        "setosa",
        "versicolor",
        "virginica"
    ]
    return {"prediction": species[prediction[0]]}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
http://127.0.0.1:8000/predict/
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
docker build -t iris-api .
docker run -d -p 8000:8000 iris-api
http://localhost:8000/predict/
docker tag iris-api <your_dockerhub_username>/iris-api
docker push <your_dockerhub_username>/iris-api
from sklearn.datasets import load_iris
data = load_iris()

