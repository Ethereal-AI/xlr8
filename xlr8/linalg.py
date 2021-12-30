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
from numpy.ctypeslib import as_array
import scipy.sparse as _spsparse
import ctypes as _ctypes

try:
    from sparse_dot_mkl._mkl_interface import (
        MKL,
        sparse_matrix_t,
        _create_mkl_sparse,
        debug_print,
        debug_timer,
        _order_mkl_handle,
        _destroy_mkl_handle,
        _type_check,
        _empty_output_check,
        _is_allowed_sparse_format,
        _check_return_value,
        _allocate_for_export,
    )
    from sparse_dot_mkl._sparse_sparse import _matmul_mkl
except:
    pass


def mkl_dot(matrix_a, matrix_b, cast=False):
    # Create intel MKL objects
    mkl_a, a_dbl = _create_mkl_sparse(matrix_a)
    mkl_b, b_dbl = _create_mkl_sparse(matrix_b)

    mkl_c = _matmul_mkl(mkl_a, mkl_b)

    python_c = export_mkl(mkl_c)
    return python_c


def export_mkl(csr_mkl_handle):
    """
    Export a MKL sparse handle of CSR or CSC type
    :param csr_mkl_handle: Handle for the MKL internal representation
    :type csr_mkl_handle: sparse_matrix_t
    :return: Sparse matrix in scipy format
    :rtype: scipy.spmatrix
    """

    out_func = MKL._mkl_sparse_d_export_csr
    sp_matrix_constructor = _spsparse.csr_matrix

    # Allocate for output
    ordering, nrows, ncols, indptrb, indptren, indices, data = _allocate_for_export(
        True
    )
    final_dtype = np.float64

    ret_val = out_func(
        csr_mkl_handle,
        _ctypes.byref(ordering),
        _ctypes.byref(nrows),
        _ctypes.byref(ncols),
        _ctypes.byref(indptrb),
        _ctypes.byref(indptren),
        _ctypes.byref(indices),
        _ctypes.byref(data),
    )

    # Get matrix dims
    ncols, nrows = ncols.value, nrows.value

    # Get the index dimension
    index_dim = nrows

    # Construct a numpy array and add 0 to first position for scipy.sparse's 3-array indexing
    indptrb = as_array(indptrb, shape=(index_dim,))
    indptren = as_array(indptren, shape=(index_dim,))

    indptren = np.insert(indptren, 0, indptrb[0])
    nnz = indptren[-1] - indptrb[0]

    # Construct numpy arrays from data pointer and from indicies pointer
    data = np.array(as_array(data, shape=(nnz,)), copy=True)
    indices = np.array(as_array(indices, shape=(nnz,)), copy=True)

    # Pack and return the matrix
    return sp_matrix_constructor((data, indices, indptren), shape=(nrows, ncols))


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


def sparse_dot_product(
    a, b, *, dense_output=False, use_float=False, approx_size=1.0, blas="default"
):
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
            dot_product = mkl_dot
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
        if _spsparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            a_b = dot_product(a, b_2d)
            a_b = a_b.reshape(a.shape[0], *b_.shape[1:])
        elif _spsparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            a_b = dot_product(a_2d, b)
            a_b = a_b.reshape(*a.shape[:-1], b.shape[1])
        else:
            a_b = np.dot(a, b)
    else:
        a_b = dot_product(a, b)

    if (
        _spsparse.issparse(a)
        and _spsparse.issparse(b)
        and dense_output
        and hasattr(a_b, "toarray")
    ):
        return a_b.toarray()
    return a_b
