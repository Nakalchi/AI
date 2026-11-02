import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = r"C:\Users\Mandar\Downloads\ml-100k\u.data"

column_names = ["user_id", "item_id", "rating", "timestamp"]

data = pd.read_csv(
    file_path,
    sep="\t",
    names=column_names
)

data["user_id"] = data["user_id"] - 1
data["item_id"] = data["item_id"] - 1

num_users = int(data["user_id"].max()) + 1
num_items = int(data["item_id"].max()) + 1

print("Number of Users:", num_users)
print("Number of Items:", num_items)

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

def create_tf_dataset(df):
    users = tf.constant(
        df["user_id"].values.astype(np.int32)
    )

    items = tf.constant(
        df["item_id"].values.astype(np.int32)
    )

    ratings = tf.constant(
        df["rating"].values.astype(np.float32)
    )

    return tf.data.Dataset.from_tensor_slices(
        ((users, items), ratings)
    ).shuffle(1024).batch(128)

train_dataset = create_tf_dataset(train_data)
test_dataset = create_tf_dataset(test_data)

class MatrixFactorizationModel(tf.keras.Model):

    def __init__(self, num_users, num_items, embedding_dim=20):
        super().__init__()

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=int(num_users),
            output_dim=embedding_dim
        )

        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=int(num_items),
            output_dim=embedding_dim
        )

    def call(self, inputs):
        users, items = inputs

        user_vector = self.user_embedding(users)
        item_vector = self.item_embedding(items)

        return tf.reduce_sum(
            user_vector * item_vector,
            axis=1
        )

model = MatrixFactorizationModel(
    num_users,
    num_items
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01
    ),
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=2
)

test_loss, test_mae = model.evaluate(
    test_dataset,
    verbose=2
)

print("\nTest Loss:", round(test_loss, 4))
print("Test MAE:", round(test_mae, 4))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

def recommend(user_id, top_k=5):

    user_vector = tf.constant(
        [user_id] * num_items,
        dtype=tf.int32
    )

    item_vector = tf.constant(
        np.arange(num_items),
        dtype=tf.int32
    )

    predictions = model.predict(
        (user_vector, item_vector),
        verbose=0
    ).flatten()

    top_items = np.argsort(predictions)[-top_k:][::-1]

    return top_items

user_id = 0

recommended_items = recommend(user_id)

print("\nTop Recommended Item IDs:")
print(recommended_items)
