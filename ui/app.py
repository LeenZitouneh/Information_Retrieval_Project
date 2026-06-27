# =========================================
# IR System User Interface
# =========================================


import sys
import os

from services.query.query_refinement import QueryRefiner


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


import streamlit as st



from services.search_service import SearchService


from services.model_loader import (

    tfidf,

    bm25,

    embedding,

    embedding_vector,

    hybrid,

    documents,

    doc_ids

)



# =========================================
# Create Search Service
# =========================================


search_service = SearchService(

    tfidf=tfidf,

    bm25=bm25,

    embedding=embedding,

    embedding_vector=embedding_vector,

    hybrid=hybrid,

    documents=documents,

    doc_ids=doc_ids

)



# =========================================
# Interface
# =========================================


st.title(
    "touche2020 Information Retrieval System"
)


query = st.text_input(

    "Enter Query"

)

refiner = QueryRefiner()

query = refiner.refine_query(query)

st.write("Refined query:", query)


model = st.selectbox(

    "Choose Retrieval Model",

    [

        "TF-IDF",

        "BM25",

        "Embedding",
        
        "Embedding + FAISS",

        "Hybrid Serial",

        "Hybrid Parallel"

    ]

)

# =========================================
# BM25 Parameters
# =========================================


if model == "BM25":


    st.subheader(
        "BM25 Parameters"
    )


    k1 = st.slider(

        "k1",

        0.1,

        3.0,

        1.5

    )


    b = st.slider(

        "b",

        0.0,

        1.0,

        0.75

    )



# =========================================
# Search
# =========================================


if st.button("Search"):


    if query.strip() == "":


        st.warning(
            "Enter Query First"
        )


    else:


        # Update BM25 parameters

        if model == "BM25":


            bm25.update_parameters(

                k1,

                b,

                documents

            )



        results = search_service.search(

            query,

            model=model,

        )



        st.subheader(
            "Search Results"
        )



        # ============================
        # Display Results
        # ============================


        for rank, result in enumerate(

            results,

            start=1

        ):


            st.write(

                f"### Rank {rank}"

            )

            # حالة TF-IDF / BM25

            if isinstance(result, tuple):


                idx, score = result

                doc_id = doc_ids[idx]

                st.write(

                    "Document ID:",

                    doc_id

                )

                st.write(

                    "Score:",

                    float(score)

                )

                
            # حالة Embedding / Hybrid

            else:

                doc_id = result["doc_id"]
                
                st.write(

                    "Document ID:",

                    result["doc_id"]

                )


                st.write(

                    "Score:",

                    result["score"]

                )

# ==========================
    # Show Document Text
    # ==========================




            doc_index = doc_ids.index(doc_id)


            st.write(

                "Document Text:"

            )


            st.write(

                documents[doc_index]  #[:500]

            )


            st.divider()