import random
import numpy as np
from jugador import Jugador
from grid import Grid


class Entorno():
    """Entorno con el que interactuara la IA."""

    def __init__(self, grid=Grid, factorDescuento=0):
        self.grid = grid
        self.grilla = self.grid.get_grilla()
        self.jugador = self.grid.get_jugador()
        self.factorDescuento = factorDescuento
        self.done = False

    def get_recompensa(self):
        recompensa = -1
        last_pos_x = self.grid.get_last_pos_x()
        last_pos_y = self.grid.get_last_pos_y()
        pos_x = self.jugador.get_posicion_x()
        pos_y = self.jugador.get_posicion_y()
        pos_x, pos_y = self.get_movimiento(pos_x, pos_y)
        try:
            desface_hacia_abajo = pos_x < 0 or pos_y < 0
            desface_hacia_arriba = pos_x > last_pos_x or pos_y > last_pos_y
            if desface_hacia_abajo or desface_hacia_arriba:
                raise IndexError()
            if self.grilla[pos_y][pos_x] == 3:
                recompensa = 1
            elif self.grilla[pos_y][pos_x] == 1:
                recompensa = -0.001
            elif self.grilla[pos_y][pos_x] == -1:
                recompensa = -1
            return recompensa
        except IndexError:
            return recompensa

    def get_movimiento(self, pos_x, pos_y):
        if self.accion == 0:
            pos_x -= 1
        elif self.accion == 1:
            pos_x += 1
        elif self.accion == 2:
            pos_y -= 1
        elif self.accion == 3:
            pos_y += 1
        return pos_x, pos_y

    def set_accion(self, accion=0):
        self.accion = accion

    def realizar_movimiento(self):
        last_pos_x = self.grid.get_last_pos_x()
        last_pos_y = self.grid.get_last_pos_y()
        pos_x = self.jugador.get_posicion_x()
        pos_y = self.jugador.get_posicion_y()
        pos_x, pos_y = self.get_movimiento(pos_x, pos_y)
        desface_hacia_abajo = pos_x < 0 or pos_y < 0
        desface_hacia_arriba = pos_x > last_pos_x or pos_y > last_pos_y
        if not (desface_hacia_abajo or desface_hacia_arriba):
            self.jugador.set_posicion_x(pos_x)
            self.jugador.set_posicion_y(pos_y)

    def se_termino(self):
        pos_x = self.jugador.get_posicion_x()
        pos_y = self.jugador.get_posicion_y()
        if self.grilla[pos_y][pos_x] == 3:
            self.done = True

    def actuar(self, accion):
        self.set_accion(accion)
        recompensa = self.get_recompensa()
        self.realizar_movimiento()
        self.grid.update_grid()
        self.se_termino()
        return recompensa, self.done

    def get_accion(self, table):
        randomFloat = random.random()
        if randomFloat < self.factorDescuento:
            self.accion = random.randint(0, 3)
            return self.accion
        else:
            return self.get_accion_predict(table)

    def get_accion_predict(self, table):
        pos_x = self.jugador.get_posicion_x()
        pos_y = self.jugador.get_posicion_y()
        return np.argmax(table[pos_y][pos_x])
