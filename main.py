import os
import numpy as np
from keras import Input
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape, GRU
import tensorflow as tf

from tqdm import trange, tqdm

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

with open("data_final.wav", mode="rb") as f:
    our_data = f.read()[:2_000_000]

data = []
max_len = 80

for byte in our_data:
    data.append(byte)

print(f"Num sample: {len(data)}")

train_labels = []
train_data = []


def one_hot_encode(samples):
    le_array = np.zeros((256,))
    le_array[samples] = 1
    return le_array


for i in trange(max_len, len(data)):
    train_data.append(np.array(data[i - max_len:i]).reshape(-1, 1))
    train_labels.append(one_hot_encode(data[i]))

train_data = np.array(train_data)
train_labels = np.array(train_labels)

print(train_data.shape)
print(train_labels.shape)

# Define the LSTM model
inputs = Input(shape=(max_len, 1))
x = GRU(128, return_sequences=True)(inputs)
x = GRU(64, return_sequences=True)(x)
x = GRU(64)(x)
outputs = Dense(256, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print(model.predict([[0.2]*80]))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
# Train the model
for _ in range(20):
    model.fit(train_data, train_labels, epochs=1, batch_size=256, callbacks=[tensorboard_callback])
    print(model.predict([[0.2] * 80]))
    model.save(filepath="model2")
