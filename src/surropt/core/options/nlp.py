import requests
from abc import ABC


class NLPOptions(ABC):
    # TODO: Document NLPOptions class
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
        response = requests.get(self.server_url)

        if response.status_code != 200:
            raise ValueError("Couldn't connect to the server URL provided.")
