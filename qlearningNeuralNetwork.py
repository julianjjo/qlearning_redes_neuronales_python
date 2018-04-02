import numpy as np
import time
from grid import Grid
from entorno import Entorno
from itertools import chain


def get_q_value(q_value_positive, q_value_negative):
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
    max_estados = 100
    max_training_data = 1
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 7
    size_y = 7
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)
    input_training = []

    for training in range(0, max_training_data):
        grid.get_random_grid()
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
        output_data_positive = []
        output_data_negative = []
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
                    input_data = []
                    input_train = grilla.reshape(size_y*size_x)
                    input_data.append(input_train.tolist())
                    input_data.append([accion])
                    input_data = list(chain(*input_data))
                    q_value = table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action]
                    if q_value < 0:
                        output_data_negative.append(abs(q_value))
                        output_data_positive.append(0)
                    else:
                        output_data_negative.append(0)
                        output_data_positive.append(q_value)
                    input_data = np.asarray([input_data])
                    nsamples, n = input_data.shape
                    input_data = input_data.reshape((nsamples, n))
                    input_training.append(input_data)

        print("Ejecutar Qlearning Resultado")
        time.sleep(1)
        recompensaEpisodio = 0
        grid.set_grilla(grid.get_initial_grid())
        jugador.reset_to_inital_post()
        for i in range(0, max_estados):
            entorno = Entorno(grid, factorDescuento)
            action = entorno.get_accion_predict(table)
            recompensa, done = entorno.actuar()
            print(grid)
            time.sleep(1)
            recompensaEpisodio = recompensaEpisodio + recompensa
            if done is True:
                break
        print("Recompensa: ", recompensaEpisodio)

        output_data_positive = np.asarray(output_data_positive)
        output_data_negative = np.asarray(output_data_negative)
        input_training = np.asarray(input_training)
        nsamples, nx, ny = input_training.shape
        input_training = input_training.reshape((nsamples,nx*ny))
        output_data_positive = output_data_positive.ravel()
        output_data_negative = output_data_negative.ravel()
        print(table)
        print(output_data_positive)
        print(output_data_negative)
        print("Entrenar Red Neuronal")
        time.sleep(1)
        from sklearn.neural_network import MLPRegressor

        modelPostive = MLPRegressor(
            max_iter=30000000,
            hidden_layer_sizes=(100, 100, 100),
            learning_rate_init=0.00001,
        )
        modelNegative = MLPRegressor(
            max_iter=30000000,
            hidden_layer_sizes=(100, 100, 100),
            learning_rate_init=0.00001,
        )
        modelPostive.fit(input_training, output_data_positive)
        modelNegative.fit(input_training, output_data_negative)

    print("Probar Red Neuronal")

    time.sleep(1)
    recompensaPrueba = 0
    grid.set_grilla(grid.get_initial_grid())
    jugador.reset_to_inital_post()
    for i in range(0, max_estados):
        acciones = []
        for accion in range(0, 4):
            grilla = grid.get_grilla()
            input_value = []
            input_grilla = grilla.reshape(-1)
            input_value.append(input_grilla.tolist())
            input_value.append([accion])
            input_value = list(chain(*input_value))
            input_value = np.asarray([input_value])
            nsamples, n = input_value.shape
            input_value = input_value.reshape((nsamples, n))
            q_value_positive = modelPostive.predict(input_value)
            q_value_negative = modelNegative.predict(input_value)
            acciones.append(get_q_value(q_value_positive, q_value_negative))
        acciones = np.asarray(acciones)
        print(acciones)
        print(grid)
        time.sleep(1)
        accion = np.argmax(acciones)
        entorno = Entorno(grid)
        entorno.set_accion(accion)
        recompensa, done = entorno.actuar()
        recompensaPrueba += recompensa
        if done is True:
            break

    print("Recompensa: ", recompensaPrueba)


if __name__ == '__main__':
    main()
