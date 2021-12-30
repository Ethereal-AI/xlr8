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
from scipy.sparse import issparse

try:
    from sparse_dot_mkl import dot_product_mkl
except:
    pass


def uniform_approximation(a, b, c, d):
    """Creates uniformly approximate matrices of a and b, c and d.
    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    c : {ndarray, sparse matrix}
    d : {ndarray, sparse matrix}
    Returns
    -------
    c : {ndarray, sparse matrix}
    d : {ndarray, sparse matrix}
    """
    # Pick rows from a and corresponding column from b uniformly random
    n = a.shape[1]
    s = c.shape[1]
    p_each = 1.0 / n  # Since uniform all row/col have equal probability
    for t in range(0, s):
        # Pick a random row and column independently with replacement
        i_t = np.random.randint(0, n)
        c[:, t] = a[i_t, :]
        d[t, :] = b[:, i_t]

    # Apply uniform scaling
    scaling = np.sqrt(s * p_each)
    c /= scaling
    d /= scaling
    return c, d


def sparse_dot_product(a, b, *, dense_output=False, use_float=False, approx_size=1.0, blas="default"):
    """Dot product that handle the sparse matrix case correctly.
    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.
    use_float : bool, default=True
        When False, function uses default numpy datatype.
        When True, function converts ``a`` and ``b`` to float.
    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if blas != "mkl":
        dot_product = np.matmul
        convert_array = True
    elif blas == "mkl":
        try:
            dot_product = dot_product_mkl
            convert_array = False
        except:
            dot_product = np.matmul
            convert_array = True

    if convert_array == True:
        if not isinstance(a, np.ndarray):
            a = a.toarray()
        if not isinstance(b, np.ndarray):
            b = b.toarray()

    if use_float == True:
        a = a.astype(np.float32)
        b = b.astype(np.float32)

    if approx_size != 1.0:
        approx_dim = int(a.shape[1] * approx_size)
        c = np.zeros([a.shape[0], approx_dim], dtype=a.dtype)
        d = np.zeros([approx_dim, b.shape[1]], dtype=b.dtype)
        a, b = uniform_approximation(a, b, c, d)

    if a.ndim > 2 or b.ndim > 2:
        if issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            a_b = dot_product(a, b_2d)
            a_b = a_b.reshape(a.shape[0], *b_.shape[1:])
        elif issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            a_b = dot_product(a_2d, b)
            a_b = a_b.reshape(*a.shape[:-1], b.shape[1])
        else:
            a_b = np.dot(a, b)
    else:
        a_b = dot_product(a, b)

    if issparse(a) and issparse(b) and dense_output and hasattr(a_b, "toarray"):
        return a_b.toarray()
    return a_b
