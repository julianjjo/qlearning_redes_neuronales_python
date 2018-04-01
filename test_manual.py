from grid import Grid
from entorno import Entorno
from jugador import Jugador
from tesoro import Tesoro


def main():
    size_x = 5
    size_y = 7
    grid = Grid(size_x, size_y)
    jugador = grid.get_jugador()
    grid.get_random_grid()
    print(jugador.get_posicion_y())
    print(jugador.get_posicion_x())
    print(grid)
    entorno = Entorno(grid, 0.8)
    accion = entorno.get_accion()
    recompensa = entorno.actuar(accion)
    print(jugador.get_posicion_y())
    print(jugador.get_posicion_x())
    print(recompensa)
    print(grid)


if __name__ == '__main__':
    main()
