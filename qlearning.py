import numpy as np
import time
from grid import Grid
from entorno import Entorno



def main():
    minLearningRate = 0.1
    maxLearningRate = 1.0
    factorDescuento = 0.8
    episodios = 200000
    max_estados = 50
    learningRate = np.linspace(minLearningRate, maxLearningRate, episodios)
    size_x = 7
    size_y = 7
    table = np.empty([size_y, size_x, 4])
    table.fill(0)
    grid = Grid(size_x, size_y)
    grid.get_random_grid()
    entorno = Entorno(grid, factorDescuento)
    jugador = grid.get_jugador()
    pos_x = jugador.get_posicion_x()
    pos_y = jugador.get_posicion_y()
    maximaRecompensa = -1000
    for episodio in range(0, episodios):
        recompensaEpisodio = 0
        grid.set_grilla(grid.get_initial_grid())
        jugador.set_posicion_x(pos_x)
        jugador.set_posicion_y(pos_y)
        for i in range(0, max_estados):
            action = entorno.get_accion(table)
            recompensa, done = entorno.actuar(action)
            if done == True:
                table[jugador.get_posicion_y()][jugador.get_posicion_x()][action] = recompensa;
            else:
                table[jugador.get_posicion_y()][jugador.get_posicion_x()][action] += learningRate[episodio] * (recompensa + factorDescuento * np.amax(table[jugador.get_posicion_y()][jugador.get_posicion_x()]) - table[jugador.get_posicion_y()][jugador.get_posicion_x()][action]);
            recompensaEpisodio += recompensa
            if done == True:
                break
        print("Episodio: {} Recompensa: {}".format(episodio, recompensaEpisodio))
        if recompensaEpisodio > maximaRecompensa:
            maximaRecompensa = recompensaEpisodio
    print("Recompensa Maxima: ",maximaRecompensa)
    np.save('q_learning', table)
    print(table)
    time.sleep(1)

    print("Ejecutar Qlearning Resultado")
    time.sleep(1)
    grid.set_grilla(grid.get_initial_grid())
    jugador.set_posicion_x(pos_x)
    jugador.set_posicion_y(pos_y)
    for i in range(0, max_estados):
        action = entorno.get_accion_predict(table)
        recompensa, done = entorno.actuar(action)
        recompensaEpisodio += recompensa
        print(grid)
        time.sleep(1)
        if(done):
            break
    print("Recompensa: ",recompensa)


if __name__ == '__main__':
    main()
