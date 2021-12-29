# xlr8
# Copyright (C) 2021 Ethereal AI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cos
from scipy.spatial.distance import cdist
from xlr8.similarity import cosine_similarity as xlr8_cos
import timeit


x = np.random.rand(2000, 2000)
y = np.random.rand(2000, 2000)


def sklearn_cosine():
    return sk_cos(x, y)


def scipy_cosine():
    return 1.0 - cdist(x, y, "cosine")


def xlr8_cosine(use_float=False, approx_size=1.0):
    return xlr8_cos(x, y, use_float=use_float, approx_size=approx_size)


start_time = timeit.default_timer()
sklearn_cosine()
print(f"scikit-learn cosine similarity: {timeit.default_timer() - start_time}")

start_time = timeit.default_timer()
xlr8_cosine()
print(f"xlr8 default cosine similarity: {timeit.default_timer() - start_time}")

start_time = timeit.default_timer()
xlr8_cosine(use_float=True)
print(f"xlr8 float cosine similarity: {timeit.default_timer() - start_time}")

start_time = timeit.default_timer()
xlr8_cosine(approx_size=0.75)
print(f"xlr8 approximated cosine similarity: {timeit.default_timer() - start_time}")

start_time = timeit.default_timer()
xlr8_cosine(use_float=True, approx_size=0.75)
print(f"xlr8 approximated float cosine similarity: {timeit.default_timer() - start_time}")

start_time = timeit.default_timer()
scipy_cosine()
print(f"scipy approximated float cosine similarity: {timeit.default_timer() - start_time}")
