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


def sparse_dot_product(a, b, *, dense_output=False, use_float=False):
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
    if use_float == True:
	    a = a.astype(np.float32)
	    b = b.astype(np.float32)
    
    if a.ndim > 2 or b.ndim > 2:
        if issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            a_b = a @ b_2d
            a_b = a_b.reshape(a.shape[0], *b_.shape[1:])
        elif issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            a_b = a_2d @ b
            a_b = a_b.reshape(*a.shape[:-1], b.shape[1])
        else:
            a_b = np.dot(a, b)
    else:
        a_b = a @ b

    if issparse(a) and issparse(b) and dense_output and hasattr(a_b, "toarray"):
        return a_b.toarray()
    return a_b
