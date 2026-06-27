from spellchecker import SpellChecker

from nltk.corpus import wordnet

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


    # =====================================
    # Query Expansion
    # =====================================


    def expand_query(self, query):


        words = query.lower().split()


        expanded_words = words.copy()



        for word in words:


            synonyms = []

            for syn in wordnet.synsets(word):

                for lemma in syn.lemmas():

                    synonym = lemma.name().replace("_", " ")

                    if synonym != word and synonym not in synonyms:

                        synonyms.append(synonym)

            expanded_words.extend(list(synonyms)[:3])



        return " ".join(expanded_words)
    



    def refine_query(self, query):

        corrected_query = self.correct_spelling(
            query
        )

        expanded_query = self.expand_query(
            corrected_query
        )

        return expanded_query
    