import numpy as np
import time
import pickle
from sklearn.externals import joblib
from grid import Grid
from entorno import Entorno
from keras.models import model_from_json


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
    max_estados = 30
    size_x = 4
    size_y = 5
    grid = Grid(size_x, size_y)
    # model = joblib.load("model.nw")
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    while True:
        grid.set_random_grid()
        print(grid)
        parar = input('Parar y/n: ')
        if parar == "y":
            break
    # with open('grilla', 'rb') as fp:
    #     grid = pickle.load(fp)
    recompensaPrueba = 0
    for i in range(0, max_estados):
        acciones = []
        entorno = Entorno(grid)
        for accion in range(0, 4):
            grilla = grid.get_grilla()
            input_value = []
            input_grilla = grilla.reshape(size_y*size_x)
            input_value = input_grilla.tolist()
            input_value.append(accion/10)
            input_value = np.asarray([input_value])
            q_value = model.predict(input_value)
            acciones.append(get_q_tipo(q_value))
        acciones = np.asarray(acciones)
        print(grid)
        time.sleep(1)
        print(acciones)
        accion = np.argmax(acciones)
        print(accion)
        entorno.set_accion(accion)
        recompensa, done = entorno.actuar()
        recompensaPrueba += recompensa
        if done is True:
            break

    print("Recompensa: ", recompensaPrueba)


if __name__ == '__main__':
    main()
