class SearchEngine:

    def __init__(self, retriever, documents, doc_ids):
        self.retriever = retriever
        self.documents = documents
        self.doc_ids = doc_ids


    def search(self, query, top_k=5):

        results = self.retriever.search(
            query,
            top_k
        )

        output = []

        for idx, score in results:

            output.append({
                "doc_id": self.doc_ids[idx],
                "score": float(score)
            })

        return output