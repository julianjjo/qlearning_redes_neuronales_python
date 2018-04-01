import unittest
import numpy as np
from grid import Grid
from entorno import Entorno

class TestGridMethods(unittest.TestCase):

    def test_actuar_recompesa_tesoro(self):
        size_x = 7
        size_y = 7
        grid = Grid(size_x, size_y)
        jugador = grid.get_jugador()
        tesoro = grid.get_tesoro()
        grid.get_random_grid()
        tesoro.set_posicion_y(0)
        tesoro.set_posicion_x(6)
        grid.update_grid()
        jugador.set_posicion_y(0)
        jugador.set_posicion_x(5)
        grid.update_grid()
        entorno = Entorno(grid)
        entorno.set_accion(accion=1)
        recompensa, done = entorno.actuar()
        self.assertEqual(recompensa, 1)

    def test_actuar_recompesa_vacio(self):
        size_x = 7
        size_y = 7
        grid = Grid(size_x, size_y)
        jugador = grid.get_jugador()
        grid.get_random_grid()
        jugador.set_posicion_y(0)
        jugador.set_posicion_x(6)
        grid.update_grid()
        jugador.set_posicion_y(0)
        jugador.set_posicion_x(5)
        grid.update_grid()
        entorno = Entorno(grid)
        entorno.set_accion(accion=1)
        recompensa, done = entorno.actuar()
        self.assertEqual(recompensa, -0.001)

    def test_actuar_done(self):
        size_x = 7
        size_y = 7
        grid = Grid(size_x, size_y)
        jugador = grid.get_jugador()
        tesoro = grid.get_tesoro()
        grid.get_random_grid()
        tesoro.set_posicion_y(0)
        tesoro.set_posicion_x(6)
        grid.update_grid()
        jugador.set_posicion_y(0)
        jugador.set_posicion_x(5)
        grid.update_grid()
        entorno = Entorno(grid, 0.8)
        entorno.set_accion(accion=2)
        recompensa, done = entorno.actuar()
        self.assertTrue(done)

    def test_actuar_done_tesoro(self):
        size_x = 7
        size_y = 7
        grid = Grid(size_x, size_y)
        jugador = grid.get_jugador()
        tesoro = grid.get_tesoro()
        grid.get_random_grid()
        tesoro.set_posicion_y(0)
        tesoro.set_posicion_x(6)
        grid.update_grid()
        jugador.set_posicion_y(0)
        jugador.set_posicion_x(5)
        grid.update_grid()
        entorno = Entorno(grid)
        entorno.set_accion(accion=1)
        recompensa, done = entorno.actuar()
        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
