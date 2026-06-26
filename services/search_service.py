# =========================================
# Search Service
# Responsible for connecting all IR services
# =========================================


from services.query.query_processor import QueryProcessor
from services.query.query_refinement import QueryRefiner
from services.preprocessing.text_preprocessor import preprocess_text
from services.ranking.ranker import Ranker



class SearchService:



    def __init__(

        self,

        tfidf,

        bm25,

        embedding,

        embedding_vector,

        hybrid,

        documents,

        doc_ids

    ):


        self.tfidf = tfidf

        self.bm25 = bm25

        self.embedding = embedding

        self.embedding_vector = embedding_vector

        self.hybrid = hybrid


        self.documents = documents

        self.doc_ids = doc_ids



        self.query_processor = QueryProcessor(preprocess_text)


        self.query_refiner = QueryRefiner()


        self.ranker = Ranker()





    # =====================================
    # Main Search Function
    # =====================================


    def search(

        self,

        query,

        model="TF-IDF",

        use_refinement=False

    ):



        # -----------------------------
        # Query Refinement Option
        # -----------------------------


        if use_refinement:

            query = self.query_refiner.refine_query(
                query
            )



        # -----------------------------
        # Query Processing
        # -----------------------------


        processed_query = self.query_processor.process(

            query

        )



        # -----------------------------
        # Choose Retrieval Model
        # -----------------------------


        if model == "TF-IDF":


            results = self.tfidf.search(

                processed_query,

                top_k=10

            )



        elif model == "BM25":


            results = self.bm25.search(

                processed_query,

                top_k=10

            )



        elif model == "Embedding":


            results = self.embedding.search(

                processed_query,

                top_k=10

            )


        elif model == "Embedding + FAISS":

            results = self.embedding_vector.search(

                processed_query,

                top_k=10

            )

        elif model == "Hybrid Parallel":


            results = self.hybrid.parallel_search(

                processed_query,

                self.tfidf,
                
                self.doc_ids

            )



        elif model == "Hybrid Serial":


            results = self.hybrid.serial_search(

                processed_query,

                top_k=10

            )



        else:


            raise ValueError(
                "Unknown Model"
            )



        return results