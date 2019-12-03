import requests
from requests.exceptions import ConnectionError
from abc import ABC


class NLPOptions(ABC):
    """Base (abstract) class for setting the NLP solver options.

    All the NLP solvers settings have to be handled through this class and 
    interfaced in the `optimize_nlp` function.
    """
    @property
    def name(self):
        """The solver name."""
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value
        else:
            raise ValueError("'name' has to be a string.")

    # -------------------------------------------------------------------------

    def __init__(self, name: str):
        self.name = name


class IpOptOptions(NLPOptions):
    """Local installation options for IpOpt solver.

    tol : float
        Desired convergence tolerance (relative). Determines the convergence
        tolerance for the algorithm. The algorithm terminates successfully, if
        the (scaled) NLP error becomes smaller than this value. Default is 1e-8

    max_iter : int
        Maximum number of iterations. The algorithm terminates with an error
        message if the number of iterations exceeded this number. The
        valid range for this integer option is 0 < max iter < +inf and its
        default value is 3000.

    con_tol : int
        Desired threshold for the constraint violation. Absolute tolerance on
        the constraint violation. Successful termination requires that the
        max-norm of the (unscaled) constraint violation is less than this
        threshold. The valid range for this real option is
        0 < constr viol tol < +inf and its default value is 1e-4.
    """
    @property
    def tol(self):
        """Desired convergence tolerance (relative)."""
        return self._tol

    @tol.setter
    def tol(self, value):
        if isinstance(value, float):
            if value > 0:
                self._tol = value
            else:
                raise ValueError("'tol' can't be zero or negative value.")
        else:
            raise ValueError("'tol' has to be a float.")

    @property
    def max_iter(self):
        """Maximum number of iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if isinstance(value, int):
            if value > 0:
                self._max_iter = value
            else:
                raise ValueError("'max_iter' can't be zero or negative value.")
        else:
            raise ValueError("'max_iter' has to be a integer.")

    @property
    def con_tol(self):
        """Desired threshold for the constraint violation."""
        return self._con_tol

    @con_tol.setter
    def con_tol(self, value):
        if isinstance(value, float):
            if value > 0:
                self._con_tol = value
            else:
                raise ValueError("'con_tol' can't be zero or negative value.")
        else:
            raise ValueError("'con_tol' has to be a float.")
    # -------------------------------------------------------------------------

    def __init__(self, name: str, tol: float = 1e-8, max_iter: int = 3000,
                 con_tol: float = 1e-4):
        super().__init__(name)

        self.tol = tol
        self.max_iter = max_iter
        self.con_tol = con_tol


class DockerNLPOptions(NLPOptions):
    """Solver options for when using IpOpt solver that is interfaced with a
    flask application inside a Docker container or WSL enviroment.

    Parameters
    ----------
    name : str
        Custom name to the `NLPOptions` object (i.e. just a identifier, not
        used as check anywhere else.)

    server_url : str
        Ip address of the docker server.

    tol : float
        Desired convergence tolerance (relative). Determines the convergence
        tolerance for the algorithm. The algorithm terminates successfully, if
        the (scaled) NLP error becomes smaller than this value. Default is 1e-8

    max_iter : int
        Maximum number of iterations. The algorithm terminates with an error
        message if the number of iterations exceeded this number. The
        valid range for this integer option is 0 < max iter < +inf and its
        default value is 3000.

    con_tol : int
        Desired threshold for the constraint violation. Absolute tolerance on
        the constraint violation. Successful termination requires that the
        max-norm of the (unscaled) constraint violation is less than this
        threshold. The valid range for this real option is
        0 < constr viol tol < +inf and its default value is 1e-4.
    """
    @property
    def server_url(self):
        """Ip address of the docker server."""
        return self._server_url

    @server_url.setter
    def server_url(self, value):
        if isinstance(value, str):
            self._server_url = value
        else:
            raise ValueError("'server_url' has to be a string.")

    @property
    def tol(self):
        """Desired convergence tolerance (relative)."""
        return self._tol

    @tol.setter
    def tol(self, value):
        if isinstance(value, float):
            if value > 0:
                self._tol = value
            else:
                raise ValueError("'tol' can't be zero or negative value.")
        else:
            raise ValueError("'tol' has to be a float.")

    @property
    def max_iter(self):
        """Maximum number of iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if isinstance(value, int):
            if value > 0:
                self._max_iter = value
            else:
                raise ValueError("'max_iter' can't be zero or negative value.")
        else:
            raise ValueError("'max_iter' has to be a integer.")

    @property
    def con_tol(self):
        """Desired threshold for the constraint violation."""
        return self._con_tol

    @con_tol.setter
    def con_tol(self, value):
        if isinstance(value, float):
            if value > 0:
                self._con_tol = value
            else:
                raise ValueError("'con_tol' can't be zero or negative value.")
        else:
            raise ValueError("'con_tol' has to be a float.")

    # -------------------------------------------------------------------------
    def __init__(self, name: str, server_url: str, tol: float = 1e-8,
                 max_iter: int = 3000, con_tol: float = 1e-4):
        super().__init__(name)

        self.server_url = server_url
        self.tol = tol
        self.max_iter = max_iter
        self.con_tol = con_tol

        self.test_connection()

    def test_connection(self):

        try:
            response = requests.get(self.server_url)
        except ConnectionError:
            raise ValueError("Couldn't connect to the server URL provided. "
                             "Make sure that the optimization server is "
                             "online and communicating properly.")
        else:
            if response.status_code != 200:
                raise ValueError("Connection to the server established. "
                                 "However, the server is unresponsive.")
