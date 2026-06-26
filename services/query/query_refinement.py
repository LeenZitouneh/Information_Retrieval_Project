from spellchecker import SpellChecker


# =====================================
# Query Refinement Service
# Improves user queries before retrieval
# =====================================


class QueryRefiner:





    def correct_spelling(self, query):

        words = query.lower().split()

        corrected_words = []

        for word in words:

            corrected_word = self.spell.correction(word)

            if corrected_word:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)





    def __init__(self):

        # Simple synonym dictionary
        # Can be expanded later

        self.spell = SpellChecker()

        self.synonyms = {


            "car": [
                "automobile",
                "vehicle"
            ],


            "food": [
                "cuisine",
                "dish"
            ],


            "war": [
                "conflict",
                "battle"
            ],


            "museum": [
                "gallery",
                "exhibition"
            ],


            "architecture": [
                "building",
                "design"
            ]

        }



    # =====================================
    # Query Expansion
    # =====================================


    def expand_query(self, query):


        words = query.lower().split()


        expanded_words = words.copy()



        for word in words:


            if word in self.synonyms:


                expanded_words.extend(
                    self.synonyms[word]
                )



        return " ".join(expanded_words)
    



    def refine_query(self, query):

        corrected_query = self.correct_spelling(
            query
        )

        expanded_query = self.expand_query(
            corrected_query
        )

        return expanded_query