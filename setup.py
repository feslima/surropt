from setuptools import setup, find_packages

setup(name="surropt", packages=find_packages())

# links to set a development environment so tests discovery works properly
# http://doc.pytest.org/en/latest/goodpractices.html#install-package-with-pip
# https://quantecon.org/wiki-py-conda-dev-env/

# to install the package in development mode (in the setup.py file directory)
# python setup.py develop

# To uninstall the development package:
# python setup.py develop --uninstall
