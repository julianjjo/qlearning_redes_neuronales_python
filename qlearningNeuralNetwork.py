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
    factorDescuento = 0.8
    episodios = 120000
    max_estados = 30
    max_training_data = 400
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 7
    size_y = 7
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)
    input_training = []
    output_data_positive = []
    output_data_negative = []
    for training in range(0, max_training_data):
        grid.set_random_grid()
        entorno = Entorno(grid, factorDescuento)
        jugador = grid.get_jugador()
        maximaRecompensa = -1000
        for episodio in range(0, episodios):
            time_to_break_out_of_a = False
            recompensaEpisodio = 0
            grid.set_grilla(grid.get_initial_grid())
            jugador.reset_to_inital_post()
            for i in range(0, max_estados):
                entorno = Entorno(grid, factorDescuento)
                entorno.set_accion_q_learning(table)
                action = entorno.get_accion()
                recompensa, done = entorno.actuar()
                if done is True:
                    table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action] = recompensa;
                else:
                    table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action] += learningRate[episodio] * (recompensa + factorDescuento * np.amax(table[jugador.get_posicion_y()][jugador.get_posicion_x()]) - table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action]);
                recompensaEpisodio = recompensaEpisodio + recompensa
                if done is True:
                    break
            if time_to_break_out_of_a is True:
                break
            print("Episodio: {} Recompensa: {}".format(episodio, recompensaEpisodio))
            if recompensaEpisodio > maximaRecompensa:
                maximaRecompensa = recompensaEpisodio
        print("Recompensa Maxima: ", maximaRecompensa)
        time.sleep(1)
        grid.set_grilla(grid.get_initial_grid())
        jugador.reset_to_inital_post()
        for post_x in range(0, size_y):
            for post_y in range(0, size_x):
                for accion in range(0, 4):
                    jugador.set_posicion_prev_x(post_x)
                    jugador.set_posicion_prev_y(post_y)
                    jugador.set_posicion_x(post_x)
                    jugador.set_posicion_y(post_y)
                    grid.update_grid()
                    entorno = Entorno(grid)
                    entorno.set_accion(accion)
                    recompensa, done = entorno.actuar()
                    grilla = grid.get_grilla()
                    input_train = grilla.reshape(size_y*size_x)
                    input_train = input_train.tolist()
                    input_train.append(accion/10)
                    q_value = table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action]
                    if q_value < 0:
                        output_data_negative.append(abs(q_value))
                        output_data_positive.append(0)
                    else:
                        output_data_negative.append(0)
                        output_data_positive.append(q_value)
                    input_training.insert(len(input_training), input_train)

    output_data_positive = np.asarray([output_data_positive])
    output_data_negative = np.asarray([output_data_negative])
    input_training = np.asarray(input_training)
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
        max_iter=9000000000000,
        hidden_layer_sizes=(100, 100, 100),
        learning_rate_init=0.0001,
    )
    modelNegative = MLPRegressor(
        max_iter=9000000000000,
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
