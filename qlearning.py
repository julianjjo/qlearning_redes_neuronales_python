import numpy as np
import time
from grid import Grid
from entorno import Entorno


def main():
    minLearningRate = 0.1
    maxLearningRate = 1.0
    factorDescuento = 0.8
    episodios = 10000
    max_estados = 100
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 4
    size_y = 5
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)

    while True:
        grid.set_random_grid()
        print(grid)
        parar = input('Parar y/n: ')
        if parar == "y":
            break
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
                table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action] = round(recompensa, 1);
            else:
                table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action] += round(learningRate[episodio] * (recompensa + factorDescuento * np.amax(table[jugador.get_posicion_y()][jugador.get_posicion_x()]) - table[jugador.get_posicion_prev_y()][jugador.get_posicion_prev_x()][action]), 1);
            recompensaEpisodio = recompensaEpisodio + recompensa
            if done is True:
                break
        if time_to_break_out_of_a is True:
            break
        print("Episodio: {} Recompensa: {}".format(episodio, recompensaEpisodio))
        if recompensaEpisodio > maximaRecompensa:
            maximaRecompensa = recompensaEpisodio
    print("Recompensa Maxima: ", maximaRecompensa)
    np.save('q_learning', table)
    time.sleep(1)

    print("Ejecutar Qlearning Resultado")
    time.sleep(1)
    recompensaEpisodio = 0
    grid.set_grilla(grid.get_initial_grid())
    jugador.reset_to_inital_post()
    for i in range(0, max_estados):
        entorno = Entorno(grid, factorDescuento)
        action = entorno.get_accion_predict(table)
        print(action)
        recompensa, done = entorno.actuar()
        print(grid)
        time.sleep(1)
        recompensaEpisodio = recompensaEpisodio + recompensa
        if done is True:
            break
    print("Recompensa: ", recompensaEpisodio)


if __name__ == '__main__':
    main()
