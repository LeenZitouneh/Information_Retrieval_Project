
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# Hybrid Retriever
# Serial Hybrid:
# BM25 -> Embedding Re-ranking
# =====================================


class HybridRetriever:


    def __init__(
        self,
        
        bm25_retriever,

        embedding_retriever
    ):


        # BM25 model
        self.bm25 = bm25_retriever


        # Embedding model
        self.embedding = embedding_retriever



# =====================================
# Serial Hybrid Search
# =====================================


    def serial_search(
        self,

        query,

        top_k=10,

        candidates=100
    ):


    # ---------------------------------
    # Step 1:
    # BM25 retrieves candidates
    # ---------------------------------


        bm25_results = self.bm25.search(

            query,

            top_k=candidates

        )



    # ---------------------------------
    # Step 2:
    # Extract candidate indexes
    # ---------------------------------


        candidate_ids = [

            result[0]

            for result in bm25_results

        ]



    # ---------------------------------
    # Step 3:
    # Get candidate documents text
    # ---------------------------------


        candidate_documents = [

            self.embedding.documents[idx]

            for idx in candidate_ids

        ]


        candidate_doc_ids = [

            self.embedding.doc_ids[idx]

            for idx in candidate_ids

        ]



    # ---------------------------------
    # Step 4:
    # Calculate embeddings only
    # for candidates
    # ---------------------------------


        candidate_embeddings = (

            self.embedding.model.encode(

                candidate_documents

            )

        )



    # ---------------------------------
    # Step 5:
    # Calculate similarity
    # ---------------------------------


        query_embedding = (

            self.embedding.model.encode(

                [query]

            )

        )



        scores = cosine_similarity(

            query_embedding,

            candidate_embeddings

        )[0]



        ranked = scores.argsort()[::-1]



    # ---------------------------------
    # Step 6:
    # Return final results
    # ---------------------------------


        results = [

            {

                "doc_id": candidate_doc_ids[idx],

                "score": float(scores[idx])

            }

            for idx in ranked[:top_k]

        ]


        return results

# =====================================
    # Parallel Hybrid Search
    # TF-IDF + BM25 Fusion
    # =====================================


    def parallel_search(
        self,
        query,
        tfidf_retriever,
        doc_ids,
        top_k=10,
        alpha=0.5
    ):



        # ---------------------------------
        # Get TF-IDF Results
        # ---------------------------------

        tfidf_results = tfidf_retriever.search(

            query,

            top_k

        )



        # ---------------------------------
        # Get BM25 Results
        # ---------------------------------

        bm25_results = self.bm25.search(

            query,

            top_k

        )



        # ---------------------------------
        # Normalize scores
        # ---------------------------------

        tfidf_scores = {

            doc_id: score

            for doc_id, score in tfidf_results

        }


        bm25_scores = {


            doc_ids[idx]:

            score

            for idx, score in bm25_results

        }



        # ---------------------------------
        # Fusion
        # ---------------------------------


        all_docs = set(

            list(tfidf_scores.keys())

            +

            list(bm25_scores.keys())

        )



        final_scores = {}



        for doc in all_docs:


            score = (

                alpha * tfidf_scores.get(
                    doc,
                    0
                )

                +

                (1-alpha) * bm25_scores.get(
                    doc,
                    0
                )

            )


            final_scores[doc] = score




        # ---------------------------------
        # Sort Results
        # ---------------------------------


        ranked = sorted(

            final_scores.items(),

            key=lambda x:x[1],

            reverse=True

        )



        return [

            {

                "doc_id":doc,

                "score":score

            }

            for doc,score in ranked[:top_k]

        ]