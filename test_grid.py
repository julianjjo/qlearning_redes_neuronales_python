import unittest
import numpy as np
from grid import Grid
from jugador import Jugador
from tesoro import Tesoro


class TestGridMethods(unittest.TestCase):

    def test_get_grilla_vacia(self):
        size_x = 2
        size_y = 2
        grid = Grid(size_x, size_y)
        grid.generate_grilla_vacia()
        grilla = grid.get_grilla()
        expect = np.array([[0, 0], [0, 0]])
        self.assertTrue(np.array_equal(grilla, expect))

    def test_set_random_jugador(self):
        size_x = 2
        size_y = 2
        grid = Grid(size_x, size_y)
        grid.generate_grilla_vacia()
        jugador_new = Jugador()
        grid.set_jugador(jugador_new)
        grid.random_posicion_jugador()
        grilla = grid.get_grilla()
        self.assertEqual(np.count_nonzero(grilla == 2), 1)

    def test_set_random_tesoro(self):
        size_x = 2
        size_y = 2
        grid = Grid(size_x, size_y)
        grid.generate_grilla_vacia()
        tesoro_new = Tesoro()
        grid.set_tesoro(tesoro_new)
        grid.random_posicion_tesoro()
        grilla = grid.get_grilla()
        self.assertEqual(np.count_nonzero(grilla == 3), 1)


if __name__ == '__main__':
    unittest.main()
