# Surropt
Surrogate optimization toolbox for time consuming models

# Installation
To install the module in develop moode, first you need to setup an environment with the following packages:

- SciPy >= 1.2.0
- Numpy >= 1.15.0
- pyDOE2 >= 1.2
- pydace >= 0.1.1
- cyipopt >= 1.0.3

Having these installed, open a terminal window, navigate to the folder where the setup.py file is located and execute the following command:
```
$python setup.py develop
```

After this you are ready to use the package via python command line.

# Usage

## Optimization server
### Server environment installation
Make sure WSL Ubuntu is installed (**NOT UBUNTU LTS, IT HAS TO BE PURE UBUNTU**) in your system.

Make sure that Anaconda is installed in your WSL system.

Open a WSL terminal and navigate to folder **tests_/resources/ipopt_server/**.

Install the server by executing the following line in the WSL terminal:

```
conda env create -f ipopt_server.yaml
```

### Starting the server
Each time you are going to perform a optimization through Caballero's algorithm using the `DockerNLPOptions` as NLP solver, you have to start the server manually. To do so, execute the following steps:

1. Open a WSL terminal and navigate to folder **tests_/resources/ipopt_server/**
2. Activate the `ipopt_server` conda environment
3. Start the server by typing in the WSL terminal: ```$python server.py```
4. If everything is fine, you should see that a flask server is initialized
5. To make sure that the server is good to go, open a browser window and type `localhost:5000`. You should see the following message on your browser: "*Hey! I'm running from Flask in a Docker container!*". If so, you can close the browser tab (**do not close the WSL terminal while performing the optimization!**) and proceed normally.

## Optimization procedure
1. Start the optimization server.

2. See file *test_evap.py* in folder **tests_/surropt/caballero/**. You can run it to see how a simple example of usage the Caballero procedure is done.