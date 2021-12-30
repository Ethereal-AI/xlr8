# xlr8

Fast cosine similarity for Python

### Installing the package
1. Clone the repository.<br>
2. Run `pip install -e .` inside the local repository.<br>

### Optional installation
If you wish to leverage `xlr8`'s further speedup on large matrix multiplications, you may install the following:
1. First, `sparse_dot` via `pip install sparse-dot-mkl`.
2. Then, `Intel MKL` via `conda install -c intel mkl`.

### Usage

Using the cosine similarity function is straightforward.
```python
from xlr8.similarity import cosine_similarity
import numpy as np

x = np.random.rand(1000,1000)
y = np.random.rand(1000,1000)

cosine_similarity(x,y)
```

### Benchmarking

To benchmark xlr8's different modes for performing cosine similarity, run `python tests/benchmark.py <dimension size>` from the main directory of the repository.<br>

Here's an example of running the benchmark with A and B set to sizes of 10,000 x 10,000:
```console
$ python benchmark.py 10000
scikit-learn cosine similarity: 14.492997799999998
xlr8 default cosine similarity: 15.422745399999997
xlr8 float cosine similarity: 9.0765971
xlr8 approximated cosine similarity: 16.5568625
xlr8 approximated float cosine similarity: 8.802123799999997
```

### Approximation

This repository implements the uniform approximate matrix multiplication method found in this [paper](http://perso.ens-lyon.fr/loris.marchal/docs-data-aware/papers/paper9.pdf) by Drineas, Kannan, and Mahoney [1].

[1] Drineas, P., Kannan, R., & Mahoney, M. W. (2006). Fast Monte Carlo algorithms for matrices I: Approximating matrix multiplication. SIAM Journal on Computing, 36(1), 132-157.