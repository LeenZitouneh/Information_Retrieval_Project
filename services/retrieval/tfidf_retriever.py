from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.indexing import inverted_index


class TFIDFRetriever:

    def __init__(self, inverted_index):
        self.vectorizer = TfidfVectorizer()
        self.inverted_index = inverted_index    

    def fit(self, documents):

        self.tfidf_matrix = self.vectorizer.fit_transform(   #tf idf for every word
            documents
        )

    def search(self, query, top_k=5):

        query_vector = self.vectorizer.transform(
            [query]
        )

    # ==========================
    # Candidate Documents
    # ==========================

        candidate_indices = set()

        for term in query.split():

            if term in self.inverted_index:

                candidate_indices.update(
                    self.inverted_index[term]
                )

    # إذا لم توجد أي وثيقة
        if not candidate_indices:

            print("No candidate documents found.")

            return []

        candidate_indices = list(candidate_indices)

    # ==========================
    # Test
    # ==========================

        print(
            f"Searching in {len(candidate_indices)} candidate documents "
            f"instead of {self.tfidf_matrix.shape[0]}"
            )
    # ==========================
    # Similarity
    # ==========================


        candidate_matrix = self.tfidf_matrix[
            candidate_indices
        ]

        similarities = cosine_similarity(
            query_vector,
            candidate_matrix
        ).flatten()

        ranked = similarities.argsort()[::-1]

        return [
            (candidate_indices[idx], similarities[idx])
            for idx in ranked[:top_k]
        ]