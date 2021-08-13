# -*- coding: utf-8 -*-
# generated with poetry2setup > setup.py
# see: https://github.com/abersheeran/poetry2setup
# https://github.com/python-poetry/poetry/issues/761
from setuptools import setup

package_dir = \
    {'': 'src'}

packages = \
    ['surropt',
     'surropt.caballero',
     'surropt.core',
     'surropt.core.nlp',
     'surropt.core.options',
     'surropt.core.procedures',
     'surropt.core.utils',
     'surropt.utils']

package_data = \
    {'': ['*']}

install_requires = \
    ['colorama>=0.4.1,<0.5.0',
     'numpy>=1.15.0,<2.0.0',
     'pydace>=0.1.3,<0.2.0',
     'pydoe2>=1.2.1,<2.0.0',
     'requests>=2.20.1,<3.0.0',
     'scipy>=1.2.0,<2.0.0']

setup_kwargs = {
    'name': 'surropt',
    'version': '0.0.12',
    'description': 'Surrogate optimization toolbox for time consuming models',
    'long_description': '# Surropt\nSurrogate optimization toolbox for time consuming models\n\n# Installation\nTo install the module in develop moode, first you need to setup an environment with the following packages:\n\n- SciPy >= 1.2.0\n- Numpy >= 1.15.0\n- pyDOE2 >= 1.2\n- pydace >= 0.1.1\n- cyipopt >= 1.0.3\n\nHaving these installed, open a terminal window, navigate to the folder where the setup.py file is located and execute the following command:\n```\n$python setup.py install\n```\n\nAfter this you are ready to use the package via python command line.\n\n# Usage\n\n## Optimization server\n### Server environment installation\nMake sure WSL Ubuntu is installed (**NOT UBUNTU LTS, IT HAS TO BE PURE UBUNTU**) in your system.\n\nMake sure that Anaconda is installed in your WSL system.\n\nOpen a WSL terminal and navigate to folder **tests_/resources/ipopt_server/**.\n\nInstall the server by executing the following line in the WSL terminal:\n\n```\nconda env create -f ipopt_server.yaml\n```\n\n### Starting the server\nEach time you are going to perform a optimization through Caballero\'s algorithm using the `DockerNLPOptions` as NLP solver, you have to start the server manually. To do so, execute the following steps:\n\n1. Open a WSL terminal and navigate to folder **tests_/resources/ipopt_server/**\n2. Activate the `ipopt_server` conda environment\n3. Start the server by typing in the WSL terminal: ```$python server.py```\n4. If everything is fine, you should see that a flask server is initialized\n5. To make sure that the server is good to go, open a browser window and type `localhost:5000`. You should see the following message on your browser: "*Hey! I\'m running from Flask in a Docker container!*". If so, you can close the browser tab (**do not close the WSL terminal while performing the optimization!**) and proceed normally.\n\n## Optimization procedure\n1. Start the optimization server.\n\n2. See file *test_evap.py* in folder **tests_/surropt/caballero/**. You can run it to see how a simple example of usage the Caballero procedure is done.',
    'author': 'Felipe Souza Lima',
    'author_email': 'feslima93@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/feslima/surropt',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.5,<4.0',
}


setup(**setup_kwargs)
