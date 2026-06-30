from rank_bm25 import BM25Okapi



class BM25Retriever:


    def __init__(

        self,

        inverted_index,

        k1=1.5,

        b=0.75

    ):

        self.inverted_index = inverted_index

        # BM25 Parameters
        
        self.k1 = k1  #TF Saturation.

        self.b = b    #length of doc -> 1 no long


        self.bm25 = None

        self.documents = None



    # ===================================
    # Build BM25 Index
    # ===================================


    def fit(self, documents):


        self.documents = documents

        tokenized_docs = [

            doc.split()

            for doc in documents

        ]


        self.bm25 = BM25Okapi(

            tokenized_docs,

            k1=self.k1,

            b=self.b

        )
        #عدد الوثائق التي تحتوي على كلمة معينة مرة واحدة على الأقل



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


    # ==========================
    # Candidate Documents
    # ==========================

        candidate_indices = set()

        for term in tokenized_query:

            if term in self.inverted_index:

                candidate_indices.update(

                    self.inverted_index[term]

                )

        if not candidate_indices:

            print("No candidate documents found.")

            return []

        candidate_indices = list(candidate_indices)

        print(

            f"Searching in {len(candidate_indices)} candidate documents "
            f"instead of {len(self.documents)}"

        )


    # ==========================
    # BM25 Scores
    # =========================

        scores = self.bm25.get_scores(

            tokenized_query

        )


        candidate_scores = [

            (idx, scores[idx])

            for idx in candidate_indices

        ]


        candidate_scores.sort(

             key=lambda x: x[1],

            reverse=True

        )



        return candidate_scores[:top_k]