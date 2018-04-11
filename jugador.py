from objeto_grilla import ObjetoGrilla


class Jugador(ObjetoGrilla):
    """Define el objetp Jugador."""

    def __init__(self, pos_x=0, pos_y=0):
        super().__init__(pos_x, pos_y)
        self.set_tipo(0.2)

    def get_posicion_prev_x(self):
        return self.pos_prev_x

    def get_posicion_prev_y(self):
        return self.pos_prev_y

    def set_posicion_prev_x(self, pos_prev_x=0):
        self.pos_prev_x = pos_prev_x

    def set_posicion_prev_y(self, pos_prev_y=0):
        self.pos_prev_y = pos_prev_y

    def get_posicion_initial_x(self):
        return self.pos_initial_x

    def get_posicion_initial_y(self):
        return self.pos_initial_y

    def set_posicion_initial_x(self, pos_initial_x=0):
        self.pos_initial_x = pos_initial_x

    def set_posicion_initial_y(self, pos_initial_y=0):
        self.pos_initial_y = pos_initial_y

    def reset_to_inital_post(self):
        self.pos_x = self.pos_initial_x
        self.pos_y = self.pos_initial_y
        self.pos_prev_x = self.pos_initial_x
        self.pos_prev_y = self.pos_initial_y

    def is_inital_post(self):
        if self.pos_initial_x == self.pos_x and self.pos_initial_y == self.pos_y:
            return True
        return False
