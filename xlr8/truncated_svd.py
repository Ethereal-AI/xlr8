"""Truncated SVD for sparse matrices, aka latent semantic analysis (LSA).
Original code from scikit-learn.
Edited by Ethereal AI to further modify parameters of TruncatedSVD.
"""

# Author: Lars Buitinck
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Michael Becker <mike@beckerfuffle.com>
#         Ethereal AI
# License: 3-clause BSD.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot, svd_flip, randomized_range_finder
from sklearn.utils.sparsefuncs import mean_variance_axis


__all__ = ["TruncatedSVD"]


class TruncatedSVD(TransformerMixin, BaseEstimator):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with sparse matrices
    efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in :mod:`sklearn.feature_extraction.text`. In
    that context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on `X * X.T` or
    `X.T * X`, whichever is more efficient.

    Read more in the :ref:`User Guide <LSA>`.

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    n_iter : int, default=5
        Number of iterations for randomized SVD solver. Not used by ARPACK. The
        default is larger than the default in
        :func:`~sklearn.utils.extmath.randomized_svd` to handle sparse
        matrices that may have large slowly decaying spectrum.

    random_state : int, RandomState instance or None, default=None
        Used during randomized svd. Pass an int for reproducible results across
        multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors of the input data.

    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

    singular_values_ : ndarray od shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    FactorAnalysis : A simple linear generative model with
        Gaussian latent variables.
    IncrementalPCA : Incremental principal components analysis.
    KernelPCA : Kernel Principal component analysis.
    NMF : Non-Negative Matrix Factorization.
    PCA : Principal component analysis.

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_dense = np.random.rand(100, 100)
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = csr_matrix(X_dense)
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)
    TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> print(svd.explained_variance_ratio_)
    [0.0157... 0.0512... 0.0499... 0.0479... 0.0453...]
    >>> print(svd.explained_variance_ratio_.sum())
    0.2102...
    >>> print(svd.singular_values_)
    [35.2410...  4.5981...   4.5420...  4.4486...  4.3288...]
    """

    def __init__(
        self,
        n_components=2,
        *,
        n_oversamples=10,
        n_iter=5,
        random_state=None,
        tol=0.0,
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def randomized_svd(
        self,
        M,
        n_components,
        *,
        n_oversamples=10,
        n_iter="auto",
        power_iteration_normalizer="auto",
        transpose="auto",
        flip_sign=True,
        random_state="warn",
    ):
        """Computes a truncated randomized SVD.
        This method solves the fixed-rank approximation problem described in the
        Halko et al paper (problem (1.5), p5).
        Parameters
        ----------
        M : {ndarray, sparse matrix}
            Matrix to decompose.
        n_components : int
            Number of singular values and vectors to extract.
        n_oversamples : int, default=10
            Additional number of random vectors to sample the range of M so as
            to ensure proper conditioning. The total number of random vectors
            used to find the range of M is n_components + n_oversamples. Smaller
            number can improve speed but can negatively impact the quality of
            approximation of singular vectors and singular values. Users might wish
            to increase this parameter up to `2*k - n_components` where k is the
            effective rank, for large matrices, noisy problems, matrices with
            slowly decaying spectrums, or to increase precision accuracy. See Halko
            et al (pages 5, 23 and 26).
        n_iter : int or 'auto', default='auto'
            Number of power iterations. It can be used to deal with very noisy
            problems. When 'auto', it is set to 4, unless `n_components` is small
            (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
            This improves precision with few components. Note that in general
            users should rather increase `n_oversamples` before increasing `n_iter`
            as the principle of the randomized method is to avoid usage of these
            more costly power iterations steps. When `n_components` is equal
            or greater to the effective matrix rank and the spectrum does not
            present a slow decay, `n_iter=0` or `1` should even work fine in theory
            (see Halko et al paper, page 9).
            .. versionchanged:: 0.18
        power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
            Whether the power iterations are normalized with step-by-step
            QR factorization (the slowest but most accurate), 'none'
            (the fastest but numerically unstable when `n_iter` is large, e.g.
            typically 5 or larger), or 'LU' factorization (numerically stable
            but can lose slightly in accuracy). The 'auto' mode applies no
            normalization if `n_iter` <= 2 and switches to LU otherwise.
            .. versionadded:: 0.18
        transpose : bool or 'auto', default='auto'
            Whether the algorithm should be applied to M.T instead of M. The
            result should approximately be the same. The 'auto' mode will
            trigger the transposition if M.shape[1] > M.shape[0] since this
            implementation of randomized SVD tend to be a little faster in that
            case.
            .. versionchanged:: 0.18
        flip_sign : bool, default=True
            The output of a singular value decomposition is only unique up to a
            permutation of the signs of the singular vectors. If `flip_sign` is
            set to `True`, the sign ambiguity is resolved by making the largest
            loadings for each component in the left singular vectors positive.
        random_state : int, RandomState instance or None, default='warn'
            The seed of the pseudo random number generator to use when
            shuffling the data, i.e. getting the random vectors to initialize
            the algorithm. Pass an int for reproducible results across multiple
            function calls. See :term:`Glossary <random_state>`.
            .. versionchanged:: 1.2
                The previous behavior (`random_state=0`) is deprecated, and
                from v1.2 the default value will be `random_state=None`. Set
                the value of `random_state` explicitly to suppress the deprecation
                warning.
        Notes
        -----
        This algorithm finds a (usually very good) approximate truncated
        singular value decomposition using randomization to speed up the
        computations. It is particularly fast on large matrices on which
        you wish to extract only a small number of components. In order to
        obtain further speed up, `n_iter` can be set <=2 (at the cost of
        loss of precision). To increase the precision it is recommended to
        increase `n_oversamples`, up to `2*k-n_components` where k is the
        effective rank. Usually, `n_components` is chosen to be greater than k
        so increasing `n_oversamples` up to `n_components` should be enough.
        References
        ----------
        * Finding structure with randomness: Stochastic algorithms for constructing
          approximate matrix decompositions (Algorithm 4.3)
          Halko, et al., 2009 https://arxiv.org/abs/0909.4061
        * A randomized algorithm for the decomposition of matrices
          Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
        * An implementation of a randomized algorithm for principal component
          analysis
          A. Szlam et al. 2014
        """

        if random_state == "warn":
            random_state = 0

        random_state = check_random_state(random_state)
        n_random = n_components + n_oversamples
        n_samples, n_features = M.shape

        if transpose == "auto":
            transpose = n_samples < n_features
        if transpose:
            # this implementation is a bit faster with smaller shape[1]
            M = M.T

        Q = randomized_range_finder(
            M,
            size=n_random,
            n_iter=n_iter,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )

        # project M to the (k + p) dimensional space using the basis vectors
        B = safe_sparse_dot(Q.T, M)

        # compute the SVD on the thin matrix: (k + p) wide
        Uhat, s, Vt = linalg.svd(B, full_matrices=False, check_finite=False, overwrite_a=True)

        del B
        U = np.dot(Q, Uhat)

        if flip_sign:
            if not transpose:
                U, Vt = svd_flip(U, Vt)
            else:
                # In case of transpose u_based_decision=false
                # to actually flip based on u and not v.
                U, Vt = svd_flip(U, Vt, u_based_decision=False)

        if transpose:
            # transpose back the results according to the input convention
            return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
        else:
            return U[:, :n_components], s[:n_components], Vt[:n_components, :]

    def fit(self, X, y=None):
        """Fit model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        # X = self._validate_data(X, accept_sparse=["csr", "csc"], ensure_min_features=2)
        random_state = check_random_state(self.random_state)

        k = self.n_components
        n_features = X.shape[1]
        if k >= n_features:
            raise ValueError(
                "n_components must be < n_features; got %d >= %d" % (k, n_features)
            )
        U, Sigma, VT = self.randomized_svd(
            X, self.n_components, n_oversamples=self.n_oversamples, power_iteration_normalizer="none", n_iter=self.n_iter, random_state=random_state
        )

        self.components_ = VT

        # As a result of the SVD approximation error on X ~ U @ Sigma @ V.T,
        # X @ V is not the same as U @ Sigma
        X_transformed = safe_sparse_dot(X, self.components_.T)

        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        # X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)
        return safe_sparse_dot(X, self.components_.T)

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}