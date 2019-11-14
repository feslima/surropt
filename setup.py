from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name="surropt",
      version='0.0.1',
      description="Surrogate optimization toolbox for time consuming models",
      author="Felipe Souza Lima",
      author_email='feslima93@gmail.com',
      url='https://github.com/feslima/surropt',
      license='Apache License 2.0',
      packages=find_packages(exclude=['tests_']),
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords="surrogate optimization, infill criteria optimization, blackbox optimization",
      install_requires=['numpy>=1.15', 'scipy>=1.2.0', 'pydace>=0.1.1',
                        'pydoe2>=1.2.1'],
      python_requires='>=3.5',
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering"
      ])
