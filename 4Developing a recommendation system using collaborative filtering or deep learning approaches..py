import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = "ml-100k/u.data"

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

data = pd.read_csv(
    file_path,
    sep='\t',
    names=column_names
)

data['user_id'] -= 1
data['item_id'] -= 1

num_users = data['user_id'].max() + 1
num_items = data['item_id'].max() + 1

print(f"Number of users: {num_users}, Number of items: {num_items}")

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

def create_tf_dataset(df):
    users = tf.constant(
        df['user_id'].values,
        dtype=tf.int32
    )

    items = tf.constant(
        df['item_id'].values,
        dtype=tf.int32
    )

    ratings = tf.constant(
        df['rating'].values,
        dtype=tf.float32
    )

    return tf.data.Dataset.from_tensor_slices(
        ((users, items), ratings)
    ).shuffle(1024).batch(32)

train_dataset = create_tf_dataset(train_data)
test_dataset = create_tf_dataset(test_data)

class MatrixFactorizationModel(tf.keras.Model):

    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()

        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_dim
        )

        self.item_embedding = tf.keras.layers.Embedding(
            num_items,
            embedding_dim
        )

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[0])
        item_vector = self.item_embedding(inputs[1])

        dot_product = tf.reduce_sum(
            user_vector * item_vector,
            axis=1
        )

        return dot_product

model = MatrixFactorizationModel(
    num_users,
    num_items
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01
    ),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

test_loss, test_mae = model.evaluate(
    test_dataset
)

print(
    f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}"
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('MAE over Epochs')

plt.show()

def recommend(user_id, top_k=5):
    user_vector = tf.constant(
        [user_id] * num_items,
        dtype=tf.int32
    )

    item_vector = tf.constant(
        list(range(num_items)),
        dtype=tf.int32
    )

    predictions = model.predict(
        (user_vector, item_vector),
        verbose=0
    )

    top_items = np.argsort(
        -predictions
    )[:top_k]

    return top_items

user_id = 0

recommended_items = recommend(user_id)

print("Recommended Items:", recommended_items)
