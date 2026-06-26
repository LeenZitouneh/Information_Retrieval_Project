# =========================================
# Model Loader Service
# Load all IR components once
# =========================================


from beir.datasets.data_loader import GenericDataLoader


from services.preprocessing.text_preprocessor import preprocess_text

from services.indexing.inverted_index import build_inverted_index

from services.retrieval.tfidf_retriever import TFIDFRetriever

from services.retrieval.bm25_retriever import BM25Retriever

from services.retrieval.embedding_retriever import EmbeddingRetriever

from services.retrieval.hybrid_retriever import HybridRetriever



print("Loading DBPedia...")


corpus, queries, qrels = GenericDataLoader(
    data_folder="data/dbpedia/dbpedia-entity"
).load(split="test")



# نفس التصغير الذي عملناه

small_corpus = dict(
    list(corpus.items())[:1000]
)



documents = []

doc_ids = []



for doc_id, doc in small_corpus.items():


    text = preprocess_text(

        doc["title"] + " " + doc["text"]

    )


    documents.append(text)

    doc_ids.append(doc_id)



# ===============================
# TF-IDF
# ===============================


tfidf = TFIDFRetriever()

tfidf.fit(
    documents
)



# ===============================
# BM25
# ===============================


bm25 = BM25Retriever()

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



# ===============================
# Hybrid
# ===============================


hybrid = HybridRetriever(

    bm25_retriever=bm25,

    embedding_retriever=embedding

)