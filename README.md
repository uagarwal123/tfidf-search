# tfidf-search

Using a combination of spaCy and TF-IDF vectorizer to generate a retreival system over my study notes that can be interacted with using the command line interface.

I preprocessed my notes using regex pattern matching and spaCy's methods like lemmatization, stop words removal and lowercasing to reduce vocabulary size.
Since retrieving what course a query belongs to is not very useful, I artifically created chunks of each course into lectures so as to narrow down where exactly the query's answer is located. I used TF-IDF vectorization, which downweights words that are common across the corpus, to produce representations of the text with my Vocabulary ranging from 1Gram to 5Gram tokens and then used cosine similarity to assess relevance against a query.

I finally implemented the ability to interact with the system using the command line interface.
