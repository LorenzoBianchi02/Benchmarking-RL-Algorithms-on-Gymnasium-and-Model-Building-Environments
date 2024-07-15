# INSTALLATION

If you want to follow the official documentation it is found here: https://alejandrocn7.github.io/sinergym/compilation/html/pages/installation.html.
Note: the documentation is not up to date and some steps have to modified: https://github.com/ugr-sail/sinergym/issues/398

Otherwise, the following repository already contains sinergym and a compatible version of EnergyPlus.
The following steps should be sufficient to get everything working:

1. ``$ git clone https://github.com/LorenzoBianchi02/Benchmarking-RL-Algorithms-on-Gymnasium-and-Model-Building-Environments.git``

2. ``$ cd Benchmarking-RL-Algorithms-on-Gymnasium-and-Model-Building-Environments``

3. ``$ virtualenv env_sinergym --python=python3.10``

4. ``$ source env_sinergym/bin/activate``

5. ``$ pip install -e .[extras]``

6. Create the following 2 environment variables:
``PYTHONPATH="/usr/local/EnergyPlus-23-1-0"`` and ``EPLUS_PATH="/usr/local/EnergyPlus-23-1-0"``
(to create a temporary  environment variable: ``$ export PYTHONPATH="/usr/local/EnergyPlus-23-1-0"``)

8. ``$ pip install -e .``
