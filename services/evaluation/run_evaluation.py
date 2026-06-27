# =========================================
# Evaluation Runner
# Tests all retrieval models
# =========================================


from services.evaluation.evaluator import (
    evaluate,
    average_scores
)


from main import (

    tfidf,

    bm25,

    embedding,

    embedding_vector,

    hybrid,

    documents,

    doc_ids,

    queries,

    qrels

)



# ==============================
# Evaluation settings
# ==============================


NUMBER_OF_QUERIES = 100



# ==============================
# TF-IDF Evaluation
# ==============================


def evaluate_tfidf():


    scores = {}


    for query_id, query_text in list(
        queries.items()
    )[:NUMBER_OF_QUERIES]:


        results = tfidf.search(

            query_text,

            top_k=10

        )



        run = {

            query_id: {

                doc_ids[idx]: score

                for idx, score in results

            }

        }



        scores.update(

            evaluate(

                {
                    query_id:
                    qrels[query_id]
                },

                run

            )

        )



    return average_scores(scores)




# ==============================
# BM25 Evaluation
# ==============================


def evaluate_bm25():


    scores = {}



    for query_id, query_text in list(
        queries.items()
    )[:NUMBER_OF_QUERIES]:


        results = bm25.search(

            query_text,

            top_k=10

        )


        run = {


            query_id: {


                doc_ids[idx]: score


                for idx, score in results


            }

        }



        scores.update(

            evaluate(

                {
                    query_id:
                    qrels[query_id]

                },

                run

            )

        )



    return average_scores(scores)





# ==============================
# Embedding Evaluation
# ==============================


def evaluate_embedding():


    scores = {}



    for query_id, query_text in list(
        queries.items()
    )[:NUMBER_OF_QUERIES]:


        results = embedding.search(

            query_text,

            top_k=10

        )



        run = {


            query_id: {


                r["doc_id"]: r["score"]

                for r in results


            }

        }



        scores.update(

            evaluate(

                {
                    query_id:
                    qrels[query_id]

                },

                run

            )

        )


    return average_scores(scores)



# ==============================
# Embedding + Vector Evaluation
# ==============================


def evaluate_embedding_vector():


    scores = {}



    for query_id, query_text in list(
        queries.items()
    )[:NUMBER_OF_QUERIES]:


        results = embedding_vector.search(

            query_text,

            top_k=10

        )



        run = {


            query_id: {


                r["doc_id"]: r["score"]

                for r in results


            }

        }



        scores.update(

            evaluate(

                {
                    query_id:
                    qrels[query_id]

                },

                run

            )

        )


    return average_scores(scores)

# ==============================
# Hybrid Serial Evaluation
# ==============================


def evaluate_hybrid_serial():


    scores = {}



    for query_id, query_text in list(
        queries.items()
    )[:NUMBER_OF_QUERIES]:


        results = hybrid.serial_search(

            query_text,

            top_k=10

        )



        run = {


            query_id: {


                r["doc_id"]: r["score"]

                for r in results


            }

        }



        scores.update(

            evaluate(

                {
                    query_id:
                    qrels[query_id]

                },

                run

            )

        )


    return average_scores(scores)





# ==============================
# Run
# ==============================


if __name__ == "__main__":



    print("\n========== TF-IDF ==========")

    print(

        evaluate_tfidf()

    )



    print("\n========== BM25 ==========")

    print(

        evaluate_bm25()

    )



    print("\n========== Embedding ==========")

    print(

        evaluate_embedding()

    )


    print("\n========== Embedding + FAISS ==========")

    print(
        evaluate_embedding_vector()
    )


    print("\n========== Hybrid Serial ==========")

    print(

        evaluate_hybrid_serial()

    )