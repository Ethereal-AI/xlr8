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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos
from xlr8.similarity import cosine_similarity as xlr8_cos
import warnings
import timeit
import numpy as np


def query_similarity(vectorizer, docs_tfidf, query, library="sklearn"):
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
    query_tfidf = query_tfidf.toarray()
    docs_tfidf = docs_tfidf.toarray()

    if library == "sklearn":
        cosine_similarities = sk_cos(query_tfidf, docs_tfidf).flatten()
    elif library == "xlr8":
        cosine_similarities = xlr8_cos(
            query_tfidf, docs_tfidf, use_float=False, compression_rate=1.0
        ).flatten()
    else:
        warnings.warn(
            "The value for the library argument you provided does not exist. Using `sklearn` by default."
        )
        cosine_similarities = sk_cos(query_tfidf, docs_tfidf).flatten()
    return cosine_similarities


docs = fetch_20newsgroups(
    subset="train", categories=["sci.med"], shuffle=True, random_state=42
).data
vectorizer = TfidfVectorizer()
docs_tfidf = vectorizer.fit_transform(docs)
query = """
An allergy is an immune system response to a foreign substance that’s not typically harmful to your body. These foreign substances are called allergens. They can include certain foods, pollen, or pet dander.
Your immune system’s job is to keep you healthy by fighting harmful pathogens. It does this by attacking anything it thinks could put your body in danger. Depending on the allergen, this response may involve inflammation, sneezing, or a host of other symptoms.
Your immune system normally adjusts to your environment. For example, when your body encounters something like pet dander, it should realize it’s harmless. In people with dander allergies, the immune system perceives it as an outside invader threatening the body and attacks it.
Allergies are common. Several treatments can help you avoid your symptoms.
"""
print(f"Query document: {query}")

start_time = timeit.default_timer()
index_sklearn = np.argmax(query_similarity(vectorizer, docs_tfidf, query))
most_similar_sklearn = docs[index_sklearn]
print(
    f"scikit-learn document similarity speed in seconds: {timeit.default_timer() - start_time}"
)

start_time = timeit.default_timer()
index_xlr8 = np.argmax(query_similarity(vectorizer, docs_tfidf, query, library="xlr8"))
most_similar_xlr8 = docs[index_xlr8]
print(
    f"xlr8 document similarity speed in seconds: {timeit.default_timer() - start_time}"
)

print(
    f"Did scikit-learn and xlr8 find the same 'most similar document'? {index_xlr8==index_sklearn}"
)
