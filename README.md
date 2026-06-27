# Information Retrieval Project

## Project Structure

```text
C:.
│   .gitignore
│   main.py
│   
├───.vscode
│       settings.json
│       
├───data
│   ├───dbpedia
│   │   └───dbpedia-entity
│   │       │   corpus.jsonl
│   │       │   queries.jsonl
│   │       │   
│   │       └───qrels
│   │               dev.tsv
│   │               test.tsv
│   │               
│   └───touche
│       └───webis-touche2020
│           │   corpus.jsonl
│           │   queries.jsonl
│           │   
│           └───qrels
│                   test.tsv
│                   
├───models
├───notebooks
│       01_Load_Datasets.ipynb
│       02_Preprocessing.ipynb
│       03_Indexing.ipynb
│       
├───report
├───services
│   │   model_loader.py
│   │   search_service.py
│   │   
│   ├───evaluation
│   │   │   evaluator.py
│   │   │   run_evaluation.py
│   │   │   
│   │   └───__pycache__
│   │           evaluator.cpython-313.pyc
│   │           run_evaluation.cpython-313.pyc
│   │           
│   ├───indexing
│   │   │   inverted_index.py
│   │   │   save_index.py
│   │   │   
│   │   └───__pycache__
│   │           inverted_index.cpython-313.pyc
│   │           
│   ├───preprocessing
│   │   │   text_preprocessor.py
│   │   │   
│   │   └───__pycache__
│   │           text_preprocessor.cpython-313.pyc
│   │           
│   ├───query
│   │   │   query_processor.py
│   │   │   query_refinement.py
│   │   │   
│   │   └───__pycache__
│   │           query_processor.cpython-313.pyc
│   │           query_refinement.cpython-313.pyc
│   │           
│   ├───ranking
│   │   │   ranker.py
│   │   │   
│   │   └───__pycache__
│   │           ranker.cpython-313.pyc
│   │           
│   ├───retrieval
│   │   │   bm25_retriever.py
│   │   │   embedding_retriever.py
│   │   │   embedding_vector_retriever.py
│   │   │   hybrid_retriever.py
│   │   │   search_engine.py
│   │   │   tfidf_retriever.py
│   │   │   
│   │   └───__pycache__
│   │           bm25_retriever.cpython-313.pyc
│   │           embedding_retriever.cpython-313.pyc
│   │           embedding_vector_retriever.cpython-313.pyc
│   │           hybrid_retriever.cpython-313.pyc
│   │           search_engine.cpython-313.pyc
│   │           tfidf_retriever.cpython-313.pyc
│   │           
│   └───__pycache__
│           model_loader.cpython-313.pyc
│           search_service.cpython-313.pyc
│           
├───ui
│       app.py
│       
└───__pycache__
        main.cpython-313.pyc
```