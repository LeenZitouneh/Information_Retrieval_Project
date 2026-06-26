from rank_bm25 import BM25Okapi



class BM25Retriever:


    def __init__(

        self,

        k1=1.5,

        b=0.75

    ):


        # BM25 Parameters

        self.k1 = k1

        self.b = b


        self.bm25 = None



    # ===================================
    # Build BM25 Index
    # ===================================


    def fit(self, documents):


        tokenized_docs = [

            doc.split()

            for doc in documents

        ]


        self.bm25 = BM25Okapi(

            tokenized_docs,

            k1=self.k1,

            b=self.b

        )



    # ===================================
    # Update Parameters From UI
    # ===================================


    def update_parameters(

        self,

        k1,

        b,

        documents

    ):


        self.k1 = k1

        self.b = b


        self.fit(

            documents

        )



    # ===================================
    # Search
    # ===================================


    def search(

        self,

        query,

        top_k=10

    ):


        tokenized_query = query.split()



        scores = self.bm25.get_scores(

            tokenized_query

        )



        ranked = scores.argsort()[::-1]



        return [

            (idx, scores[idx])

            for idx in ranked[:top_k]

        ]