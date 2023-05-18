from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    # Initialize an empty numpy array to store the document vectors
    corpus_vectors = np.zeros((len(corpus), num_features), dtype=np.float32)
    
    # Iterate through each document in the corpus
    for i, document in enumerate(corpus):
        document_vector = np.zeros(num_features, dtype=np.float32)
        num_words = 0
        
        # Iterate through each word in the document
        for word in document:
            if word in model.wv:
                # If the word is in the Word2Vec model's vocabulary, add its vector to the document vector
                document_vector += model.wv[word]
                num_words += 1
        
        # Average the document vector by dividing it by the number of words in the document
        if num_words > 0:
            document_vector /= num_words
        
        # Store the document vector in the corpus vectors array
        corpus_vectors[i] = document_vector
    
    return corpus_vectors