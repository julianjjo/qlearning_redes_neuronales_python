from objeto_grilla import ObjetoGrilla


class Jugador(ObjetoGrilla):
    """Define el objetp Jugador."""

    def __init__(self, pos_x=0, pos_y=0):
        super().__init__(pos_x, pos_y)
        self.set_value(2)
