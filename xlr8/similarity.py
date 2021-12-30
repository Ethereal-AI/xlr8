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
from xlr8.truncated_svd import TruncatedSVD
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.preprocessing._data import normalize
from xlr8.linalg import sparse_dot_product


def cosine_similarity(X, Y=None, dense_output=True, use_float=False, approx_size=1.0, compression_rate=1.0):
    """Compute cosine similarity between samples in X and Y.
    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:
        K(X, Y) = <X, Y> / (||X||*||Y||)
    On L2-normalized data, this function is equivalent to linear_kernel.
    Read more in the :ref:`User Guide <cosine_similarity>`.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
        Input data.
    Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.
    dense_output : bool, default=True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.
        .. versionadded:: 0.17
           parameter ``dense_output`` for dense output.
    use_float : bool, default=True
        When False, function uses default numpy datatype.
        When True, function converts ``a`` and ``b`` to float.
    Returns
    -------
    kernel matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    
    if compression_rate != 1.0:
        # test_cr = 0.075
        svd = TruncatedSVD(n_components=int(min(Y.shape)*compression_rate), n_oversamples=0, n_iter=0, random_state=42)
        Y = svd.fit_transform(Y)
        X = svd.transform(X)

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    kernel_matrix = sparse_dot_product(
        X_normalized,
        Y_normalized.T,
        dense_output=dense_output,
        use_float=use_float,
        approx_size=approx_size,
    )
    return kernel_matrix
