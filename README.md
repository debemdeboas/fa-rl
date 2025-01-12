# fa-rl

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pucrs-automated-planning/fa-rl/blob/main/src/Function-Approximation.ipynb)

Reference implementation for function approximation in RL

## Environment Set Up

For [Anaconda Python](https://www.anaconda.com), we recommend you create a separate environment as follows:

```sh
$ conda create -n rl python=3.9
$ conda activate rl
$ conda install --file requirements.txt
```

For pip, run the following command:

```sh
$ cd fa-rl
$ pip install -r requirements.txt
```

For pip + venv, run the following commands:

```sh
$ cd fa-rl
$ python3.9 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Then run Jupyter notebook from the `src` folder:

```sh
$ cd src
$ jupyter notebook
```
