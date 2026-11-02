import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(
    url,
    usecols=[1],
    header=0
)
data = data.values.astype("float32")
plt.plot(data)
plt.title("Airline Passengers Over Time")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()
scaler = MinMaxScaler(
    feature_range=(0, 1)
)
scaled_data = scaler.fit_transform(data)
def create_dataset(dataset, look_back=1):
    X = []
    y = []
    for i in range(
        len(dataset) - look_back
    ):
        X.append(
            dataset[
                i:(i + look_back), 0
            ]
        )
        y.append(
            dataset[
                i + look_back, 0
            ]
        )
    return (
        np.array(X),
        np.array(y)
    )
look_back = 12
X, y = create_dataset(
    scaled_data,
    look_back
)
X = X.reshape(
    (X.shape[0],
     X.shape[1],
     1)
)
train_size = int(len(X) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
model = Sequential([
    LSTM(
        50,
        activation="relu",
        input_shape=(look_back, 1)
    ),
    Dense(1)
])
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(
        X_test,
        y_test
    ),
    verbose=1
)
loss = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print(
    f"Test Loss: {loss:.4f}"
)
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(
    y_pred
)
y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)
plt.figure(figsize=(10, 5))
plt.plot(
    y_test_actual,
    label="Actual"
)
plt.plot(
    y_pred,
    label="Predicted"
)
plt.title(
    "Actual vs Predicted Airline Passengers"
)
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.legend()
plt.show()
model.save(
    "lstm_time_series.h5"
)
print("Model saved as "
    "'lstm_time_series.h5'.")
    
