# =========================================
# Model Loader Service
# Load all IR components once
# =========================================


from beir.datasets.data_loader import GenericDataLoader


from services.preprocessing.text_preprocessor import preprocess_text

from services.indexing.inverted_index import build_inverted_index

from services.retrieval.embedding_vector_retriever import EmbeddingVectorRetriever
from services.retrieval.tfidf_retriever import TFIDFRetriever

from services.retrieval.bm25_retriever import BM25Retriever

from services.retrieval.embedding_retriever import EmbeddingRetriever

from services.retrieval.hybrid_retriever import HybridRetriever



print("Loading touche2020...")


corpus, queries, qrels = GenericDataLoader(
    data_folder="data/touche/webis-touche2020"
).load(split="test")



#full_corpus = corpus
full_corpus = dict(list(corpus.items())[:5000])


documents = []

doc_ids = []



for doc_id, doc in full_corpus.items():


    text = preprocess_text(

        doc["title"] + " " + doc["text"]

    )


    documents.append(text)

    doc_ids.append(doc_id)



index = build_inverted_index(
    full_corpus,
    preprocess_text
)


print(
    "Unique Terms:",
    len(index)
)


# ===============================
# TF-IDF
# ===============================


tfidf = TFIDFRetriever(index)

tfidf.fit(
    documents
)



# ===============================
# BM25
# ===============================


bm25 = BM25Retriever(index)

bm25.fit(
    documents
)



# ===============================
# Embedding
# ===============================


embedding = EmbeddingRetriever()


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

# ===============================
# Hybrid
# ===============================


hybrid = HybridRetriever(

    bm25_retriever=bm25,

    embedding_retriever=embedding

)