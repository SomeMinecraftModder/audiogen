import keras.models
from tqdm import trange
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

# define the model
model = keras.models.load_model("model2")
max_len = 80


def sample_preds(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


starting = np.random.random(size=max_len).tolist()
with open("generated.wav", "wb") as f:
    for sample in trange(500):
        # make predictions
        x_test = np.array(starting).reshape(1, max_len, 1)
        y_pred = model.predict(x_test, verbose=0)
        # print(y_pred)
        # plt.plot(y_pred[0])
        # plt.show()
        # exit()
        y_pred = y_pred[0]
        # y_pred = int(sample_preds(y_pred, temperature=1))
        y_pred = int(np.array(y_pred).argmax())
        f.write(y_pred.to_bytes(1, byteorder="little"))
        starting = starting[1:] + [y_pred]
        # print(y_pred)
