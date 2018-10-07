import numpy as np
import time
import pickle
from sklearn.externals import joblib
from grid import Grid
from entorno import Entorno


def get_q_tipo(q_value_array):
    q_value = 0
    if(q_value_array[0][1] > q_value_array[0][0]):
        q_value = q_value_array[0][1]
    else:
        q_value = q_value_array[0][0] * -1
    return q_value

def main():
    print("Probar Red Neuronal")
    time.sleep(1)
    max_estados = 40
    size_x = 4
    size_y = 5
    grid = Grid(size_x, size_y)
    model = joblib.load("model.nw")
    while True:
        grid.set_random_grid()
        print(grid)
        parar = input('Parar y/n: ')
        if parar == "y":
            break
    # with open('grilla', 'rb') as fp:
    #     grid = pickle.load(fp)
    #     print(grid)
    recompensaPrueba = 0
    for i in range(0, max_estados):
        acciones = []
        for accion in range(0, 4):
            grilla = grid.get_grilla()
            input_value = []
            input_grilla = grilla.reshape(size_y*size_x)
            input_value = input_grilla.tolist()
            input_value.append(accion/10)
            input_value = np.asarray([input_value])
            q_value = model.predict(input_value)
            print(q_value)
            acciones.append(get_q_tipo(q_value))
        acciones = np.asarray(acciones)
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
