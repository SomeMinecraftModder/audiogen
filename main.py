import keras.models
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os

from tqdm import trange, tqdm

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

with open("realdonaldtrump.csv", mode="rb") as f:
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
model = Sequential()
model.add(LSTM(64, input_shape=(max_len, 1), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(256, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
for _ in range(20):
    model.fit(train_data, train_labels, epochs=1, batch_size=256)
    model.save(filepath="model2")
