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


x = np.random.rand(1000, 1000)
y = np.random.rand(1000, 1000)


def sklearn_cosine():
    return sk_cos(x, y)


def scipy_cosine():
    return 1.0 - cdist(x, y, "cosine")


def xlr8_cosine():
    return xlr8_cos(x, y, use_float=True)


start_time = timeit.default_timer()
print(sklearn_cosine())
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
print(scipy_cosine())
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
print(xlr8_cosine())
print(timeit.default_timer() - start_time)
