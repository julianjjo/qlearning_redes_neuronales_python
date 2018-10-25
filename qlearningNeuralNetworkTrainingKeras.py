import numpy as np
import time
import pickle
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.models import model_from_json
import matplotlib.pyplot as plt

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
    output_data = np.asarray(output_data)
    input_training = np.asarray(input_training)
    print("Entrenar Red Neuronal")
    time.sleep(1)

    # model = Sequential()
    # model.add(Dense(units=60, activation='relu', input_dim=20))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=30, activation='relu'))
    # model.add(Dense(units=4, activation='softmax'))

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    try:
        model = multi_gpu_model(model, cpu_merge=False)
        print("Training using multiple GPUs..")
    except:
        print("Training using single GPU or CPU..")

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])

    model.fit(input_training, output_data, epochs=180000, batch_size=100, verbose=1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
