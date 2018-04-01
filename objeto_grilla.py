
class ObjetoGrilla:
    """Esta Clase define un objeto dentro de la grilla"""

    def __init__(self, pos_x=0, pos_y=0):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_posicion_x(self):
        return self.pos_x

    def get_posicion_y(self):
        return self.pos_y

    def set_posicion_x(self, pos_x=0):
        self.pos_x = pos_x

    def set_posicion_y(self, pos_y=0):
        self.pos_y = pos_y
