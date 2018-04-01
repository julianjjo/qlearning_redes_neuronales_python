from grid import Grid
from jugador import Jugador
from tesoro import Tesoro


def main():
    size_x = 7
    size_y = 7
    grid = Grid(size_x, size_y)
    grid.get_random_grid()
    print(grid)


if __name__ == '__main__':
    main()
