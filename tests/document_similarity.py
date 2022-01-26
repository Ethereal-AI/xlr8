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
import timeit
import warnings

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos

from xlr8.similarity import cosine_similarity as xlr8_cos


def query_similarity(vectorizer, docs_tfidf, query, library="sklearn", blas="default"):
    """Compute cosine similarity between query document and all documents.
    Parameters
    ----------
    vectorizer: TfidfVectorizer
            TfIdfVectorizer model
    docs_tfidf:
            tfidf vectors for all docs
    query: str
    library: str
            Default value is ``sklearn``, can be switched to ``xlr8``
    Returns
    -------
    cosine_similarities: ndarray
            cosine similarity between query and all docs
    """

    query_tfidf = vectorizer.transform([query])

    if library == "sklearn":
        cosine_similarities = sk_cos(query_tfidf, docs_tfidf).flatten()
    elif library == "xlr8":
        cosine_similarities = xlr8_cos(
            query_tfidf, docs_tfidf, use_float=False, compression_rate=1.0, blas=blas
        ).flatten()
    else:
        warnings.warn(
            "The value for the library argument you provided does not exist. Using `sklearn` by default."
        )
        cosine_similarities = sk_cos(query_tfidf, docs_tfidf).flatten()
    return cosine_similarities


docs = fetch_20newsgroups(
    subset="train",
    categories=[
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ],
    shuffle=True,
    random_state=42,
).data
vectorizer = TfidfVectorizer()
docs_tfidf = vectorizer.fit_transform(docs)
query = """
An allergy is an immune system response to a foreign substance that’s not typically harmful to your body. These foreign substances are called allergens. They can include certain foods, pollen, or pet dander.
Your immune system’s job is to keep you healthy by fighting harmful pathogens. It does this by attacking anything it thinks could put your body in danger. Depending on the allergen, this response may involve inflammation, sneezing, or a host of other symptoms.
Your immune system normally adjusts to your environment. For example, when your body encounters something like pet dander, it should realize it’s harmless. In people with dander allergies, the immune system perceives it as an outside invader threatening the body and attacks it.
Allergies are common. Several treatments can help you avoid your symptoms.
"""
# print(f"Query document: {query}")

start_time = timeit.default_timer()
index_xlr8_mkl = np.argmax(
    query_similarity(vectorizer, docs_tfidf, query, library="xlr8", blas="mkl")
)
# most_similar_xlr8_mkl = docs[index_xlr8_mkl]
print(
    f"xlr8 (Intel MKL) document similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
index_sklearn = np.argmax(query_similarity(vectorizer, docs_tfidf, query))
# most_similar_sklearn = docs[index_sklearn]
print(
    f"scikit-learn document similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
index_xlr8_def = np.argmax(
    query_similarity(vectorizer, docs_tfidf, query, library="xlr8")
)
# most_similar_xlr8_def = docs[index_xlr8_def]
print(
    f"xlr8 (default BLAS) document similarity speed in seconds: {timeit.default_timer() - start_time}"
)

print(
    f"Did scikit-learn and xlr8 find the same 'most similar document'? {index_xlr8_def==index_xlr8_mkl==index_sklearn}"
)
