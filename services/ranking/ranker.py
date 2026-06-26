# =====================================
# Ranking Service
# Sorts retrieved documents by score
# =====================================


class Ranker:



    def rank(
        self,
        results,
        top_k=10
    ):


        ranked_results = sorted(

            results,

            key=lambda x: x["score"],

            reverse=True

        )


        return ranked_results[:top_k]