import numpy as np
import time
import pickle
import random
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


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

    model = MLPRegressor(
        hidden_layer_sizes=(70, 50, 50, 40, 30, 30, 21, 10, 4),
    )
    # model = joblib.load("model.nw")
    error_minimal = True
    count = 0
    while error_minimal:
        model.partial_fit(input_training, output_data)
        if count > 100:
            error_minimal = False

    score = model.score(input_training, output_data)
    print('score: {}'.format(score))
    joblib.dump(model, "model.nw")


if __name__ == '__main__':
    main()
