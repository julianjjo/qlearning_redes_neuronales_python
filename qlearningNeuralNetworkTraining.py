import numpy as np
import time
import pickle
import random
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor


def main():
    size_x = 4
    size_y = 5
    table = np.empty([size_y, size_x, 4])
    table.fill(0)

    with open('output_data', 'rb') as fp:
        output_data = pickle.load(fp)
    with open('input_training', 'rb') as fp:
        input_training = pickle.load(fp)
    data_len = len(output_data)
    output_data = np.asarray(output_data)
    input_training = np.asarray(input_training)
    print(input_training)
    print(output_data)
    print("Entrenar Red Neuronal")
    time.sleep(1)

    model = MLPRegressor(
        activation='logistic',
        solver='lbfgs',
        hidden_layer_sizes=(20, 20),
    )
    error_minimal = True
    while error_minimal:
        model.fit(input_training, output_data)
        index_rand = random.randrange(0, data_len)
        input_value = np.asarray([input_training[index_rand]])
        q_value = model.predict(input_value)
        real_q_value = np.asarray([output_data[index_rand]])
        if abs(q_value[0][0] - real_q_value[0][0]) < 0.01 and (
                abs(q_value[0][1] - real_q_value[0][1]) < 0.01):
            error_minimal = False

    joblib.dump(model, "model.nw")


if __name__ == '__main__':
    main()
