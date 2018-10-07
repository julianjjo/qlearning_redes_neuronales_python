import numpy as np
import pickle
import random
from grid import Grid
from entorno import Entorno
from pathlib import Path
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor

def main():
    minLearningRate = 0.1
    maxLearningRate = 1.0
    factorDescuento = 0.7
    episodios = 15000
    max_estados = 30
    max_training_data = 1000
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
            with open('output_data', 'rb') as fp:
                output_data = pickle.load(fp)
            with open('input_training', 'rb') as fp:
                input_training = pickle.load(fp)
    for training in range(0, max_training_data):
        input_training = []
        output_data = []
        grid.set_random_grid()
        print(grid)
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
                        q_value = round(q_value, 3)
                        if q_value <= 0:
                            value = []
                            value.append(float(abs(q_value)))
                            value.append(float(0))
                            output_data.append(value)
                        else:
                            value = []
                            value.append(float(0))
                            value.append(float(q_value))
                            output_data.append(value)
                        input_training.insert(len(input_training), input_train)
        data_len = len(output_data)
        output_data = np.asarray(output_data)
        input_training = np.asarray(input_training)
        print("Entrenar Red Neuronal")
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
