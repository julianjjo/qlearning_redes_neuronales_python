import numpy as np
import pickle
from grid import Grid
from entorno import Entorno
from pathlib import Path

def main():
    minLearningRate = 0.1
    maxLearningRate = 1.0
    factorDescuento = 0.7
    episodios = 15000
    max_estados = 30
    max_training_data = 6000
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 4
    size_y = 5
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)
    my_file = Path("input_training")
    if my_file.is_file():
        cargar = input('Cargar de Archivos y/n: ')
        if cargar == "y":
            with open('output_data_positive', 'rb') as fp:
                output_data_positive = pickle.load(fp)
            with open('output_data_negative', 'rb') as fp:
                output_data_negative = pickle.load(fp)
            with open('input_training', 'rb') as fp:
                input_training = pickle.load(fp)
        else:
            input_training = []
            output_data_positive = []
            output_data_negative = []
    else:
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
            if recompensaEpisodio > maximaRecompensa:
                maximaRecompensa = recompensaEpisodio
        print("Maxima Recompensa: {} Entrenamiento: {}".format(maximaRecompensa, training))
        grid.set_grilla(grid.get_initial_grid())
        jugador.reset_to_inital_post()
        tesoro = grid.get_tesoro()
        obstaculo = grid.get_obstaculo()
        for post_y in range(0, size_y):
            for post_x in range(0, size_x):
                grid.update_grid()
                grilla = grid.get_grilla()
                if (grilla[post_y][post_x] != tesoro.get_tipo()
                        and grilla[post_y][post_x] != obstaculo.get_tipo()):
                    for accion in range(0, 4):
                        jugador.set_posicion_prev_x(post_x)
                        jugador.set_posicion_prev_y(post_y)
                        jugador.set_posicion_x(post_x)
                        jugador.set_posicion_y(post_y)
                        grid.update_grid()
                        grilla = grid.get_grilla()
                        input_train = grilla.reshape(size_y*size_x)
                        input_train = input_train.tolist()
                        input_train.append(accion/10)
                        q_value = table[post_y][post_x][accion]
                        q_value = round(q_value, 1)
                        if q_value < 0:
                            output_data_negative.append(float(abs(q_value)))
                            output_data_positive.append(float(0))
                        else:
                            output_data_negative.append(float(0))
                            output_data_positive.append(float(q_value))
                        input_training.insert(len(input_training), input_train)
        with open('input_training', 'wb') as fp:
            pickle.dump(input_training, fp)
        with open('output_data_negative', 'wb') as fp:
            pickle.dump(output_data_negative, fp)
        with open('output_data_positive', 'wb') as fp:
            pickle.dump(output_data_positive, fp)


if __name__ == '__main__':
    main()
