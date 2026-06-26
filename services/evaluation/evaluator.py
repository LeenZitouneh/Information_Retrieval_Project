# =====================================
# Evaluation Service
# Calculates IR metrics
# =====================================


import numpy as np



def average_precision(
    retrieved_docs,
    relevant_docs
):


    hits = 0

    score = 0.0



    for i, doc in enumerate(retrieved_docs):


        if doc in relevant_docs:


            hits += 1


            score += hits / (i + 1)



    if len(relevant_docs) == 0:

        return 0.0



    return score / len(relevant_docs)





def recall_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):


    retrieved = retrieved_docs[:k]


    hits = len(
        set(retrieved)
        &
        set(relevant_docs)
    )


    if len(relevant_docs) == 0:

        return 0.0



    return hits / len(relevant_docs)





# =====================================
# Precision@10
# =====================================


def precision_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):


    retrieved = retrieved_docs[:k]


    hits = len(
        set(retrieved)
        &
        set(relevant_docs)
    )


    return hits / k





# =====================================
# nDCG@10
# =====================================


def ndcg_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):


    retrieved = retrieved_docs[:k]



    dcg = 0



    for i, doc in enumerate(retrieved):


        if doc in relevant_docs:


            dcg += (
                1 /
                np.log2(i + 2)
            )



    ideal = min(
        len(relevant_docs),
        k
    )


    idcg = sum(
        1 /
        np.log2(i + 2)

        for i in range(ideal)
    )


    if idcg == 0:

        return 0.0



    return dcg / idcg





# =====================================
# Main Evaluation Function
# =====================================


def evaluate(
    qrels,
    run
):


    results = {}



    for query_id in qrels:


        relevant_docs = set(
            qrels[query_id]
        )


        retrieved_docs = list(
            run.get(query_id, {})
            .keys()
        )



        results[query_id] = {


            "MAP":

            average_precision(
                retrieved_docs,
                relevant_docs
            ),



            "Recall@10":

            recall_at_k(
                retrieved_docs,
                relevant_docs,
                10
            ),



            "Precision@10":

            precision_at_k(
                retrieved_docs,
                relevant_docs,
                10
            ),



            "NDCG@10":

            ndcg_at_k(
                retrieved_docs,
                relevant_docs,
                10
            )


        }



    return results


# =====================================
# Average Scores
# =====================================


def average_scores(results):


    total = {

        "MAP": 0,

        "Recall@10": 0,

        "Precision@10": 0,

        "NDCG@10": 0

    }


    count = len(results)



    if count == 0:

        return total



    for query_result in results.values():


        total["MAP"] += query_result["MAP"]

        total["Recall@10"] += query_result["Recall@10"]

        total["Precision@10"] += query_result["Precision@10"]

        total["NDCG@10"] += query_result["NDCG@10"]



    for key in total:


        total[key] /= count



    return total