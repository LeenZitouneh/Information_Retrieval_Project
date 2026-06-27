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

print("Loading touche2020...")

corpus, queries, qrels = GenericDataLoader(
    data_folder="data/touche/webis-touche2020"
).load(split="test")


#full_corpus = corpus
full_corpus = dict(list(corpus.items())[:500])

print(

    "Documents used:",

    len(full_corpus)

)

# ============================
# Preprocessing + Documents
# ============================


documents = []
doc_ids = []


for doc_id, doc in full_corpus.items():

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
    full_corpus,
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

    "queries",

    "qrels"

]