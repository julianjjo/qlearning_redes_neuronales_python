import numpy as np
import random
import copy
from jugador import Jugador
from tesoro import Tesoro


class Grid():
    """Esta clase genera la grilla del el laberito"""

    def __init__(self, size_x, size_y, jugador=Jugador, tesoro=Tesoro):
        self.size_x = size_x
        self.size_y = size_y
        self.last_pos_x = self.size_x - 1
        self.last_pos_y = self.size_y - 1
        self.jugador = Jugador()
        self.tesoro = Tesoro()

    def get_grilla(self):
        return self.grilla

    def set_grilla(self, grilla):
        self.grilla = grilla

    def generate_grilla_vacia(self):
        "Este metodo genera la grilla vacia"
        self.grilla = np.empty([self.size_y, self.size_x], dtype=int)
        self.grilla.fill(0)

    def set_jugador(self, jugador_new=Jugador):
        self.jugador = jugador_new

    def get_jugador(self):
        return self.jugador

    def set_tesoro(self, tesoro_new=Tesoro):
        self.tesoro = tesoro_new

    def get_tesoro(self):
        return self.tesoro

    def get_last_pos_x(self):
        return self.last_pos_x

    def get_last_pos_y(self):
        return self.last_pos_y

    def get_initial_grid(self):
        return self.initial_grid

    def set_initial_grid(self):
        self.initial_grid = self.grilla.copy()

    def get_initial_player(self):
        return self.initial_player

    def random_posicion_jugador(self):
        "Guarda la posicion y coloca el valor guarda la posicion de Jugador"
        while True:
            pos_x = random.randint(0, (self.size_x - 1))
            pos_y = random.randint(0, (self.size_y - 1))
            if self.grilla[pos_y][pos_x] == 0:
                self.jugador.set_posicion_x(pos_x)
                self.jugador.set_posicion_y(pos_y)
                self.grilla[pos_y][pos_x] = self.jugador.get_value()
                break

    def random_posicion_tesoro(self):
        "Guarda la posicion y coloca el valor guarda la posicion del tesoro"
        while True:
            pos_x = random.randint(0, (self.size_x - 1))
            pos_y = random.randint(0, (self.size_y - 1))
            if self.grilla[pos_y][pos_x] == 0:
                self.tesoro.set_posicion_x(pos_x)
                self.tesoro.set_posicion_y(pos_y)
                self.grilla[pos_y][pos_x] = self.tesoro.get_value()
                break

    def accion_aleatoria(self, pos_x, pos_y):
        accion = random.randint(0, 3)
        before_post_x = pos_x
        before_post_y = pos_y

        if accion == 0:
            if pos_x > 0:
                pos_x -= 1
            else:
                pos_x += 1
        elif accion == 1:
            if pos_x < self.last_pos_x:
                pos_x += 1
            else:
                pos_x += 1
        elif accion == 2:
            if pos_y > 0:
                pos_y -= 1
            else:
                pos_y += 1
        elif accion == 3:
            if pos_y < self.last_pos_y:
                pos_y += 1
            else:
                pos_y += 1
        try:
            desface_hacia_abajo = pos_x < 0 or pos_y < 0
            desface_hacia_arriba = pos_x > self.last_pos_x or pos_y > self.last_pos_y
            if desface_hacia_abajo or desface_hacia_arriba:
                raise IndexError()
            pos_x = pos_x
            pos_y = pos_y
            return pos_x, pos_y
        except IndexError:
            pos_x, pos_y = self.accion_aleatoria(before_post_x, before_post_y)
        return pos_x, pos_y

    def generar_camino(self, pos_x=0, pos_y=0):
        pos_x, pos_y = self.accion_aleatoria(pos_x, pos_y)
        if self.grilla[pos_y][pos_x] == self.tesoro.get_value():
            return pos_x, pos_y
        else:
            if self.grilla[pos_y][pos_x] == 0:
                self.grilla[pos_y][pos_x] = 1
                pos_x, pos_y = self.generar_camino(pos_x, pos_y)
            else:
                pos_x, pos_y = self.generar_camino(pos_x, pos_y)
        return pos_x, pos_y

    def generar_obstaculos(self):
        maximos_obstaculos = np.count_nonzero(self.grilla == 0)
        if maximos_obstaculos > 1:
            obstaculos_random = random.randint(1, maximos_obstaculos)
        else:
            obstaculos_random = maximos_obstaculos
        for value in range(0, obstaculos_random):
            pos_x = random.randint(0, (self.size_x - 1))
            pos_y = random.randint(0, (self.size_y - 1))
            if self.grilla[pos_y][pos_x] == 0:
                self.grilla[pos_y][pos_x] = -1

    def rellenar_espacios_en_blanco(self):
        self.grilla[self.grilla == 0] = 1
        pass

    def get_random_grid(self):
        self.generate_grilla_vacia()
        self.random_posicion_jugador()
        self.random_posicion_tesoro()
        self.generar_camino(
            self.jugador.get_posicion_x(), self.jugador.get_posicion_y())
        self.generar_obstaculos()
        self.rellenar_espacios_en_blanco()
        self.set_initial_grid()
        pass

    def update_grid(self):
        pos_x = self.jugador.get_posicion_x()
        pos_y = self.jugador.get_posicion_y()
        self.grilla = self.initial_grid.copy()
        self.grilla[self.grilla == self.jugador.get_value()] = 1
        self.grilla[pos_y][pos_x] = self.jugador.get_value()

    def __str__(self):
        stringGrilla = ""
        for y in self.grilla:
            for x in y:
                if x == 1:
                    stringValue = "v"
                elif x == self.jugador.get_value():
                    stringValue = "j"
                elif x == self.tesoro.get_value():
                    stringValue = "T"
                elif x == -1:
                    stringValue = "f"
                stringGrilla = stringGrilla+stringValue+"  "
            stringGrilla = stringGrilla+"\n"
        return stringGrilla
