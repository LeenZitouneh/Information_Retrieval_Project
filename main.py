from beir.datasets.data_loader import GenericDataLoader

from services.preprocessing.text_preprocessor import preprocess_text
from services.indexing.inverted_index import build_inverted_index
from services.retrieval.embedding_vector_retriever import EmbeddingVectorRetriever
from services.retrieval.tfidf_retriever import TFIDFRetriever
from services.retrieval.search_engine import SearchEngine
from services.retrieval.bm25_retriever import BM25Retriever
from services.evaluation.evaluator import evaluate, average_scores
from services.retrieval.embedding_retriever import EmbeddingRetriever
from services.retrieval.hybrid_retriever import HybridRetriever
from services.query.query_processor import QueryProcessor
from services.query.query_refinement import QueryRefiner
from services.ranking.ranker import Ranker

print("Loading DBPedia...")

corpus_dbpedia, queries_dbpedia, qrels_dbpedia = GenericDataLoader(
    data_folder="data/dbpedia/dbpedia-entity"
).load(split="test")


# ============================
# تصغير البيانات للتجربة
# ============================
small_corpus = dict(
    list(corpus_dbpedia.items())[:1000]
)


print(
    "Documents used:",
    len(small_corpus)
)

# ============================
# Preprocessing + Documents
# ============================


documents = []
doc_ids = []


for doc_id, doc in small_corpus.items():

    text = preprocess_text(
        doc["title"] + " " + doc["text"]
    )

    documents.append(text)
    doc_ids.append(doc_id)
    


# ============================
# Inverted Index
# ============================


print("Building Inverted Index...")


index = build_inverted_index(
    small_corpus,
    preprocess_text
)


print(
    "Unique Terms:",
    len(index)
)



# ============================
# TF-IDF
# ============================


tfidf  = TFIDFRetriever()

tfidf .fit(
    documents
)



# ============================
# BM25
# ============================


bm25 = BM25Retriever()

bm25.fit(
    documents
)

# ==========================
# Embedding Representation
# ==========================

embedding = EmbeddingRetriever(
    "minilm"
)


embedding.fit(
    documents,
    doc_ids
)


# ==========================
# Embedding + Vector Representation
# ==========================
embedding_vector = EmbeddingVectorRetriever(
    model_name="minilm"
)

embedding_vector.fit(
    documents,
    doc_ids
)

# =========================================
# Hybrid Representation
# =========================================


hybrid = HybridRetriever(

    bm25_retriever=bm25,

    embedding_retriever=embedding

)



print("\nSystem Ready")


# =========================================
# Export objects for UI
# =========================================


__all__ = [

    "tfidf",

    "bm25",

    "embedding",

    "embedding_vector",

    "hybrid",

    "documents",

    "doc_ids",

    "index",

    "queries_dbpedia",

    "qrels_dbpedia"

]







"""

# query_id, query_text = next(iter(queries_dbpedia.items()))
# ============================
# Evaluation
# ============================


tfidf_scores = {}


bm25_scores = {}

# ============================
# Query Processor
# ============================

query_processor = QueryProcessor(
    preprocess_text
)

query_refiner = QueryRefiner()



for query_id, query_text in list(queries_dbpedia.items())[:100]:



# ==========================
# Query Refinement
# ==========================

    refined_query = query_refiner.refine_query(
        query_text
    )

# ==========================
# Query Processing
# ==========================


    processed_query = query_processor.process(
        refined_query
    )

# ==========================
# Evaluation
# ==========================


print("\n========== TF-IDF ==========")

print(
    average_scores(tfidf_scores)
)



print("\n========== BM25 ==========")

print(
    average_scores(bm25_scores)
)



print("\n========== Embedding ==========")


embedding = EmbeddingRetriever()


embedding.fit(
    documents,
    doc_ids
)


embedding_results = embedding.search(
    query_text,
    top_k=10
)



embedding_run = {

    query_id: {

        r["doc_id"]: r["score"]

        for r in embedding_results

    }

}



embedding_scores = evaluate(

    {
        query_id: qrels_dbpedia[query_id]
    },

    embedding_run

)



print(
    average_scores(
        embedding_scores
    )
)



print("\n==========Serial Hybrid ==========")


hybrid = HybridRetriever(

    bm25_retriever=bm25,

    embedding_retriever=embedding

)


hybrid_results = hybrid.serial_search(

    processed_query,
    
    top_k=10,

    candidates=100


)




hybrid_run = {


    query_id: {


        r["doc_id"]: r["score"]


        for r in hybrid_results


    }

}



hybrid_scores = evaluate(

    {
        query_id: qrels_dbpedia[query_id]
    },

    hybrid_run

)



print(

    average_scores(

        hybrid_scores

    )

)


print("\n==========Parallel Hybrid ==========")


hybrid = HybridRetriever(

    bm25_retriever=bm25,

    embedding_retriever=embedding

)


hybrid_results = hybrid.parallel_search(

    processed_query,
    
    tfidf ,

    doc_ids

)


hybrid_run = {


    query_id: {


        r["doc_id"]: r["score"]


        for r in hybrid_results


    }

}



hybrid_scores = evaluate(

    {
        query_id: qrels_dbpedia[query_id]
    },

    hybrid_run

)



print(

    average_scores(

        hybrid_scores

    )

)





embedding_vector = EmbeddingVectorRetriever(
    model_name="minilm"
)

embedding_vector.fit(
    documents,
    doc_ids
)

results = embedding_vector.search(
    "museum",
    top_k=5
)

for result in results:

    print(result["doc_id"])
    print(result["score"])
    print(result["document"])
    print("-" * 50)


embedding = EmbeddingRetriever("minilm")
embedding.fit(documents, doc_ids)

embedding_results = embedding.search(
    "museum",
    top_k=5
)

print("\n===== Embedding =====")

for result in embedding_results:

    print(result["doc_id"])
    print(result["score"])
    print("-" * 50)

embedding_vector = EmbeddingVectorRetriever("minilm")
embedding_vector.fit(documents, doc_ids)

vector_results = embedding_vector.search(
    "museum",
    top_k=5
)

print("\n===== Embedding + FAISS =====")

for result in vector_results:

    print(result["doc_id"])
    print(result["score"])
    print(result["document"])
    print("-" * 50)

"""