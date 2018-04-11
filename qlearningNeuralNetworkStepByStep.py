import numpy as np
import os
import time
import random
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from grid import Grid
from entorno import Entorno


def get_q_table(grilla, size_x, size_y, accion, modelPostive, modelNegative):
    q_table = []
    input_value = []
    input_grilla = grilla.reshape(size_y*size_x)
    input_value = input_grilla.tolist()
    input_value.append(accion/10)
    input_value = np.asarray([input_value])
    q_value_positive = modelPostive.predict(input_value)
    q_value_negative = modelNegative.predict(input_value)
    q_table.append(get_q_tipo(
        q_value_positive, q_value_negative))
    return q_table

def get_q_tipo(q_value_positive, q_value_negative):
    q_value = 0
    if(q_value_positive > q_value_negative):
        q_value = q_value_positive
    else:
        q_value = q_value_negative * -1
    return q_value

def main():
    minLearningRate = 0.1
    maxLearningRate = 1.0
    factorDescuento = 0.8
    episodios = 120000
    max_estados = 30
    max_training_data = 30
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 7
    size_y = 7
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)
    q_value = 0
    modelPostive = MLPRegressor(
        max_iter=800,
        hidden_layer_sizes=(100, 100, 100),
        learning_rate_init=0.0001,
    )
    modelNegative = MLPRegressor(
        max_iter=800,
        hidden_layer_sizes=(100, 100, 100),
        learning_rate_init=0.0001,
    )
    for training in range(0, max_training_data):
        grid.set_random_grid()
        entorno = Entorno(grid, factorDescuento)
        jugador = grid.get_jugador()
        maximaRecompensa = -1000
        for episodio in range(0, episodios):
            recompensaEpisodio = 0
            grid.set_grilla(grid.get_initial_grid())
            jugador.reset_to_inital_post()
            for i in range(0, max_estados):
                input_training = []
                output_data_positive = []
                output_data_negative = []
                q_table = []
                q_new_table = []
                if jugador.is_inital_post() is False and episodio != 0:
                    if random.uniform(0,1) > factorDescuento:
                        for accion in range(0, 4):
                            grilla = grid.get_grilla()
                            q_table = get_q_table(grilla, size_x, size_y, accion, modelPostive, modelNegative)
                        q_table = np.asarray(q_table)
                    else:
                        q_table = np.zeros(shape=(4))
                        accion_rand = random.randrange(0, 3)
                        accion = accion_rand / 10
                        grilla = grid.get_grilla()
                        input_value = []
                        input_grilla = grilla.reshape(size_y*size_x)
                        input_grilla = grilla.reshape(size_y*size_x)
                        input_value = input_grilla.tolist()
                        input_value.append(accion)
                        input_value = np.asarray([input_value])
                        q_value_positive = modelPostive.predict(input_value)
                        q_value_negative = modelNegative.predict(input_value)
                        q_value = get_q_tipo(
                            q_value_positive, q_value_negative)
                        q_table[accion_rand] = q_value

                    accion = np.argmax(q_table)
                    entorno = Entorno(grid)
                    entorno.set_accion(accion)
                    recompensa, done = entorno.actuar()
                    for accion in range(0, 4):
                        grilla = grid.get_grilla()
                        q_new_table = get_q_table(grilla, size_x, size_y, accion, modelPostive, modelNegative)
                    grilla = grid.get_grilla()
                    input_value = []
                    input_grilla = grilla.reshape(size_y*size_x)
                    input_grilla = grilla.reshape(size_y*size_x)
                    input_value = input_grilla.tolist()
                    input_value.append(entorno.get_accion() / 10)
                else:
                    grilla = grid.get_grilla()
                    accion_rand = random.randrange(0, 3)
                    accion = accion_rand / 10
                    entorno = Entorno(grid)
                    entorno.set_accion(accion_rand)
                    recompensa, done = entorno.actuar()
                    q_table = np.zeros(shape=(4))
                    q_new_table = np.zeros(shape=(4))
                    input_value = []
                    input_grilla = grilla.reshape(size_y*size_x)
                    input_value = input_grilla.tolist()
                    input_value.append(entorno.get_accion())
                if done is True:
                    q_value = recompensa
                    if q_value < 0:
                        output_data_negative.append(abs(q_value))
                        output_data_positive.append(0)
                    else:
                        output_data_negative.append(0)
                        output_data_positive.append(q_value)
                else:
                    q_value = learningRate[episodio] * (
                        recompensa + factorDescuento * np.amax(q_new_table) - np.amax(q_table))
                    if q_value < 0:
                        output_data_negative.append(abs(q_value))
                        output_data_positive.append(0)
                    else:
                        output_data_negative.append(0)
                        output_data_positive.append(q_value)

                input_training.insert(len(input_training), input_value)
                modelPostive.partial_fit(input_training, output_data_positive)
                modelNegative.partial_fit(input_training, output_data_negative)
                recompensaEpisodio = recompensaEpisodio + recompensa
                if done is True:
                    break
            if recompensaEpisodio > maximaRecompensa:
                maximaRecompensa = recompensaEpisodio
        print("Maxima Recompensa: {} Episodio: {}".format(
                maximaRecompensa, episodio))
        # os.system("cls")

        filehandlerPositive = open("modelPostive.nw", 'w')
        filehandlerNegative = open("modelNegative.nw", 'w')
        joblib.dump(modelPostive, "modelPostive.nw")
        joblib.dump(modelNegative, "modelNegative.nw")


if __name__ == '__main__':
    main()
