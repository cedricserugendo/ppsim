DEFAULT_PROJECTION_EPSG = 21781


class Water:
    """
    Define default value for water.
    """

    def __init__(self) -> None:
        self.cp = 4186  # Joule/kg K
        self.density = 997  # kg/m³
        self.eta = 1 * 10 ** (-3)  # Pa.s (20 °C, 1 to 100 bar)


class Substation:
    """
    Define default value for substation.
    """

    def __init__(self) -> None:
        self.init_loss_coeff = 1
        self.valve_opening = True


class Junction:
    """
    Define default value for junction.
    """

    def __init__(self) -> None:
        """
        self.tfluid_k : initial value for temperature calculations
        self.pn_bar : initial value for pressure calculation
        """
        self.pn_bar = 1.0  # bar
        self.tfluid_k = 274.15 + 5.0  # kelfin


class External:
    """
    Define default value of the external environment.
    """

    def __init__(self) -> None:
        self.text_k = 10 + 273.15  # kelvin
