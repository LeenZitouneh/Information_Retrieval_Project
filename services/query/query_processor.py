# =====================================
# Query Processing
# Responsible for processing user queries
# using the same preprocessing pipeline
# =====================================


class QueryProcessor:



    def __init__(self, preprocess_function):


        self.preprocess_function = preprocess_function



    # =====================================
    # Process Query
    # =====================================


    def process(self, query):


        processed_query = self.preprocess_function(
            query
        )


        return processed_query