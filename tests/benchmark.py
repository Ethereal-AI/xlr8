# MIT License
# xlr8
# Copyright (c) 2022 Ethereal AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
import timeit

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity as sk_cos

from xlr8.similarity import cosine_similarity as xlr8_cos

try:
    dim = int(sys.argv[1])
except:
    dim = 1000
x = np.random.rand(dim, dim)
y = np.random.rand(dim, dim)


def sklearn_cosine():
    return sk_cos(x, y)


def scipy_cosine():
    return 1.0 - cdist(x, y, "cosine")


def xlr8_cosine(use_float=False, approx_size=1.0):
    return xlr8_cos(x, y, use_float=use_float, approx_size=approx_size)


start_time = timeit.default_timer()
sklearn_cosine()
print(
    f"scikit-learn cosine similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
xlr8_cosine()
print(
    f"xlr8 default cosine similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
xlr8_cosine(use_float=True)
print(
    f"xlr8 float cosine similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
xlr8_cosine(approx_size=0.75)
print(
    f"xlr8 approximated cosine similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
xlr8_cosine(use_float=True, approx_size=0.75)
print(
    f"xlr8 approximated float cosine similarity speed in seconds: {timeit.default_timer() - start_time}"
)

# start_time = timeit.default_timer()
# scipy_cosine()
# print(f"scipy approximated float cosine similarity: {timeit.default_timer() - start_time}")
