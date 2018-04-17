import numpy as np
import time
import pickle
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from grid import Grid
from entorno import Entorno

def main():
    size_x = 4
    size_y = 5
    table = np.empty([size_y, size_x, 4])
    table.fill(0)

    with open('output_data', 'rb') as fp:
        output_data = pickle.load(fp)
    with open('input_training', 'rb') as fp:
        input_training = pickle.load(fp)
    output_data = np.asarray(output_data)
    input_training = np.asarray(input_training)
    print("Entrenar Red Neuronal")
    time.sleep(1)
    data_train = input_training
    data_train = np.concatenate((
                                data_train,
                                output_data), axis=1)

    input_train, input_test, output_train, output_test = train_test_split(
        data_train[:, 0:21], data_train[:, 21:24],
        test_size=0.30
    )

    modelPostive = MLPRegressor(
        activation='relu',
        solver='lbfgs',
        max_iter=200000,
        hidden_layer_sizes=(20, 20, 20),
    )

    modelPostive.fit(input_train, output_train)

    predict = modelPostive.predict(input_test)
    data__postive_check = pd.DataFrame(predict, columns=["predict"])
    data__postive_check["y"] = list(output_test)
    data__postive_check.set_index(["y"], drop=False, inplace=True)
    data__postive_check.sort_values(by=["y"], inplace=True)
    data__postive_check.plot()
    plt.show()

    joblib.dump(modelPostive, "model.nw")


if __name__ == '__main__':
    main()
