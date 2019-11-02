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

    # -------------------------------------------------------------------------
    def __init__(self, name: str, server_url: str):
        super().__init__(name)

        self.server_url = server_url

        self.test_connection()

    def test_connection(self):

        try:
            response = requests.get(self.server_url)
        except ConnectionError:
            raise ValueError("Couldn't connect to the server URL provided.")
        else:
            if response.status_code != 200:
                raise ValueError("Connection to the server established. "
                                 "However, the server is unresponsive.")
