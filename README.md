# xlr8

Fast cosine similarity for Python

### Installing the package
1. Clone the repository.<br>
2. Run `pip install -e .` inside the local repository.<br>

### Optional installation
If you wish to leverage `xlr8`'s further speedup on large matrix multiplications, you may install the following:
1. First, `sparse_dot` via `pip install sparse-dot-mkl`.
2. Then, `Intel MKL` via `conda install -c intel mkl`.

If a warning pops up stating that your Intel MKL version is outdated, find the old .so or .dll file to something else, then rename the updated .so or .dll to the appropriate filename. This is probably caused by your system detecting the old .so or .dll. If the problem persists, try `pip install mkl`.

### Usage

Using the cosine similarity function is straightforward.
```python
from xlr8.similarity import cosine_similarity
import numpy as np

A = np.random.rand(1000,1000)
B = np.random.rand(1000,1000)

cosine_similarity(A, B)
```

### Benchmarking

To benchmark xlr8's different modes for performing cosine similarity, run `python tests/benchmark.py <dimension size>` from the main directory of the repository.<br>

Here's an example of running the benchmark with matrices A and B set to sizes of 10,000 x 10,000:
```console
$ python tests/benchmark.py 10000
scikit-learn cosine similarity speed in seconds: 14.492997799999998
xlr8 default cosine similarity speed in seconds: 15.422745399999997
xlr8 float cosine similarity speed in seconds: 9.0765971
xlr8 approximated cosine similarity speed in seconds: 16.5568625
xlr8 approximated float cosine similarity speed in seconds: 8.802123799999997
```

### Usage in a natural language processing task

You can also test the library on a document similarity task. It is recommended to use scikit-learn's cosine_similarity for smaller number of documents.

```console
$ python document_similarity.py
xlr8 (Intel MKL) document similarity speed in seconds: 0.06536109999999873
scikit-learn document similarity speed in seconds: 0.1037038999999993
xlr8 (default BLAS) document similarity speed in seconds: 13.5424935
Did scikit-learn and xlr8 find the same 'most similar document'? True
```

### Approximation

This repository implements the uniform approximate matrix multiplication method found in this [paper](http://perso.ens-lyon.fr/loris.marchal/docs-data-aware/papers/paper9.pdf) by Drineas, Kannan, and Mahoney [1].

[1] Drineas, P., Kannan, R., & Mahoney, M. W. (2006). Fast Monte Carlo algorithms for matrices I: Approximating matrix multiplication. SIAM Journal on Computing, 36(1), 132-157.