import numpy as np

import faiss

from sentence_transformers import SentenceTransformer

from gensim.models import Word2Vec




# =====================================
# Embedding Retriever Class
# =====================================


class EmbeddingVectorRetriever:



    def __init__(self, model_name="minilm"):


        self.model_name = model_name

        self.model = None

        self.document_embeddings = None

        self.documents = []

        self.doc_ids = []

        self.index = None



        # -------------------------------
        # Select Embedding Model
        # -------------------------------


        if model_name == "minilm":


            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )



        elif model_name == "bert":


            self.model = SentenceTransformer(
                "bert-base-nli-mean-tokens"
            )



        elif model_name == "word2vec":


            self.model = None



        else:

            raise ValueError(
                "Unknown embedding model"
            )



    # =====================================
    # Fit Documents
    # =====================================


    def fit(self, documents, doc_ids):


        self.documents = documents

        self.doc_ids = doc_ids



        # -------------------------------
        # Sentence Transformer Models
        # -------------------------------


        if self.model_name in [
            "minilm",
            "bert"
        ]:


            self.document_embeddings = self.model.encode(

                documents,

                show_progress_bar=True

            )


# -------------------------------
# Build FAISS Index
# -------------------------------

            self.document_embeddings = np.array(
                self.document_embeddings,
                dtype=np.float32
            )

            dimension = self.document_embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

            faiss.normalize_L2(
                self.document_embeddings
            )

            self.index.add(
                self.document_embeddings
            )

        # -------------------------------
        # Word2Vec
        # -------------------------------


        elif self.model_name == "word2vec":



            tokenized_docs = [

                doc.split()

                for doc in documents

            ]



            self.model = Word2Vec(

                sentences=tokenized_docs,

                vector_size=100,

                window=5,

                min_count=1,

                workers=4

            )



            self.document_embeddings = np.array(

                [

                    self.document_vector(doc)

                    for doc in tokenized_docs

                ]

            )



            # -------------------------------
            # Build FAISS Index
            # -------------------------------

            self.document_embeddings = np.array(
                self.document_embeddings,
                dtype=np.float32
            )

            dimension = self.document_embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

            faiss.normalize_L2(
                self.document_embeddings
            )

            self.index.add(
                self.document_embeddings
            )

    # =====================================
    # Word2Vec Document Vector
    # =====================================


    def document_vector(self, words):


        vectors = [

            self.model.wv[word]

            for word in words

            if word in self.model.wv

        ]


        if len(vectors) == 0:

            return np.zeros(
                self.model.vector_size
            )


        return np.mean(
            vectors,
            axis=0
        )



    # =====================================
    # Search
    # =====================================


    def search(self, query, top_k=10):



        # -------------------------------
        # Encode Query
        # -------------------------------


        if self.model_name in [

            "minilm",

            "bert"

        ]:


            query_vector = self.model.encode(

                [query]

            )



        else:


            query_vector = np.array(

                [

                    self.document_vector(

                        query.split()

                    )

                ]

            )



        # -------------------------------
        # Prepare Query for FAISS
        # -------------------------------

        query_vector = np.array(
            query_vector,
            dtype=np.float32
        )

        faiss.normalize_L2(
            query_vector
        )


        # -------------------------------
        # Search in FAISS Index
        # -------------------------------

        scores, indices = self.index.search(
            query_vector,
            top_k
        )


        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append(

                {

                    "doc_id": self.doc_ids[idx],

                    "document": self.documents[idx],

                    "score": float(score)

                }

            )

        return results