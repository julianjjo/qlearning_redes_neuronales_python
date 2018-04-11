import numpy as np
import time
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from grid import Grid
from entorno import Entorno

def main():
    minLearningRate = 0.1
    maxLearningRate = 1.0
    factorDescuento = 0.7
    episodios = 120000
    max_estados = 30
    max_training_data = 6000
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 7
    size_y = 7
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)

    output_data_positive = np.load("output_data_positive")
    output_data_negative = np.load("output_data_negative")
    input_training = np.load("input_training")
    data_train_positive = input_training
    data_train_negative = input_training
    data_train_positive = np.concatenate((
                                        data_train_positive,
                                        output_data_positive.T), axis=1)
    data_train_negative = np.concatenate((
                                        data_train_negative,
                                        output_data_negative.T), axis=1)
    print("Entrenar Red Neuronal")
    time.sleep(1)

    input_positive_train, input_positive_test, output_positive_train, output_positive_test = train_test_split(
        data_train_positive[:, 0:50], data_train_positive[:, 50:51].ravel(),
        test_size=0.30
    )

    input_negative_train, input_negative_test, output_negative_train, output_negative_test = train_test_split(
        data_train_negative[:, 0:50], data_train_negative[:, 50:51].ravel(),
        test_size=0.30
    )

    modelPostive = MLPRegressor(
        max_iter=90000000000000,
        hidden_layer_sizes=(100, 100, 100),
        learning_rate_init=0.0001,
    )
    modelNegative = MLPRegressor(
        max_iter=90000000000000,
        hidden_layer_sizes=(100, 100, 100),
        learning_rate_init=0.0001,
    )

    modelPostive.fit(input_positive_train, output_positive_train)
    modelNegative.fit(input_negative_train, output_negative_train)

    filehandlerPositive = open("modelPostive.nw", 'w')
    filehandlerNegative = open("modelNegative.nw", 'w')
    joblib.dump(modelPostive, "modelPostive.nw")
    joblib.dump(modelNegative, "modelNegative.nw")

    predict_positive = modelPostive.predict(input_positive_test)
    data__postive_check = pd.DataFrame(predict_positive, columns=["predict"])
    data__postive_check["y"] = list(output_positive_test)
    data__postive_check.set_index(["y"], drop=False, inplace=True)
    data__postive_check.sort_values(by=["y"], inplace=True)
    data__postive_check.plot()
    plt.show()
    predict_negative = modelNegative.predict(input_negative_test)
    data__negative_check = pd.DataFrame(predict_negative, columns=["predict"])
    data__negative_check["y"] = list(output_negative_test)
    data__negative_check.set_index(["y"], drop=False, inplace=True)
    data__negative_check.sort_values(by=["y"], inplace=True)
    data__negative_check.plot()
    plt.show()


if __name__ == '__main__':
    main()
