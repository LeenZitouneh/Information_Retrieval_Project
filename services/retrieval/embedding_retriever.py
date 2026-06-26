import numpy as np

from sentence_transformers import SentenceTransformer

from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity



# =====================================
# Embedding Retriever Class
# =====================================


class EmbeddingRetriever:



    def __init__(self, model_name="minilm"):


        self.model_name = model_name

        self.model = None

        self.document_embeddings = None

        self.documents = []

        self.doc_ids = []



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
        # Similarity
        # -------------------------------


        scores = cosine_similarity(

            query_vector,

            self.document_embeddings

        )[0]


        ranked = scores.argsort()[::-1]


        return [

            {

                "doc_id": self.doc_ids[idx],

                "score": float(scores[idx])

            }

            for idx in ranked[:top_k]

        ]