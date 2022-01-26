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
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.preprocessing._data import normalize

from xlr8.linalg import sparse_dot_product
from xlr8.truncated_svd import TruncatedSVD


def cosine_similarity(
    X,
    Y=None,
    dense_output=True,
    use_float=False,
    approx_size=1.0,
    compression_rate=1.0,
    blas="default",
):
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
        svd = TruncatedSVD(
            n_components=int(min(Y.shape) * compression_rate),
            n_oversamples=0,
            n_iter=0,
            random_state=42,
        )
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
        blas=blas,
    )
    return kernel_matrix
