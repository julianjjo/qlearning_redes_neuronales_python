import numpy as np
import time
import pickle
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import multi_gpu_model


def main():
    size_x = 4
    size_y = 5
    maximo_n_mininal = 1
    table = np.empty([size_y, size_x, 4])
    table.fill(0)

    with open('output_data', 'rb') as fp:
        output_data = pickle.load(fp)
    with open('input_training', 'rb') as fp:
        input_training = pickle.load(fp)
    data_len = len(output_data)
    print(data_len)
    output_data = np.asarray(output_data)
    input_training = np.asarray(input_training)

    print("Entrenar Red Neuronal")
    time.sleep(1)

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=21))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=21, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    try:
        model = multi_gpu_model(model, cpu_merge=False)
        print("Training using multiple GPUs..")
    except:
        print("Training using single GPU or CPU..")

    model.compile(
        loss='mean_squared_error',
        optimizer='sgd',
        metrics=['accuracy'])
    model.fit(input_training, output_data, epochs=350, batch_size=10, verbose=1)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
