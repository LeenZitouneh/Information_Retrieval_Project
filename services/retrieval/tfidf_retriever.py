from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, documents):

        self.tfidf_matrix = self.vectorizer.fit_transform(
            documents
        )

    def search(self, query, top_k=5):

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        ranked_indices = similarities.argsort()[::-1]

        return [
            (idx, similarities[idx])
            for idx in ranked_indices[:top_k]
        ]