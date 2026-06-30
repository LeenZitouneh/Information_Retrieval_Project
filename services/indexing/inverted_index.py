from collections import defaultdict, Counter



def build_inverted_index(
        corpus,
        preprocess_text,
        min_term_freq=2
):


    inverted_index = defaultdict(set)

    term_frequency = Counter()



    # ==========================
    # First pass:
    # Count terms
    # ==========================


    documents_words = {}


    for index, (doc_id, doc) in enumerate(corpus.items()):


        text = preprocess_text(

            doc["title"]

            + " "

            + doc["text"]

        )


        words = text.split()


        documents_words[index] = words


        term_frequency.update(words)



    # ==========================
    # Second pass:
    # Build index
    # ==========================


    for index, words in documents_words.items():


        for word in words:


            if term_frequency[word] >= min_term_freq:


                inverted_index[word].add(
                    index
                )


    return inverted_index

