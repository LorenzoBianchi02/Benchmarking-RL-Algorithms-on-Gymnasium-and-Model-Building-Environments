# INSTALLATION

If you want to follow the official documentation: https://alejandrocn7.github.io/sinergym/compilation/html/pages/installation.html
Note the documentation is not up to date and the some steps have to modified: https://github.com/ugr-sail/sinergym/issues/398

Otherwise:

1. ``$ git clone https://github.com/LorenzoBianchi02/Benchmarking-RL-Algorithms-on-Gymnasium-and-Model-Building-Environments.git``

2. ``$ cd Benchmarking-RL-Algorithms-on-Gymnasium-and-Model-Building-Environments``

3. ``$ virtualenv env_sinergym --python=python3.10``

4. ``$ source env_sinergym/bin/activate``

5. ``$ pip install -e .[extras]``

6. Create the following 2 environment variables:
``PYTHONPATH="/usr/local/EnergyPlus-23-1-0" \n EPLUS_PATH="/usr/local/EnergyPlus-23-1-0"``

7. ``$ pip install -e .``