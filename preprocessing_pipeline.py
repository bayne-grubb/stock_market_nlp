import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob


def preprocessing_pipeline(text):
    #remove all punctuation, including special characters
    #not accounted for by python's list of punctuation
    table = str.maketrans(string.punctuation + "“’—",
                          ' ' * (len(string.punctuation) + 3))
    text = text.translate(table)

    #tokenize text into a list of words
    words = TextBlob(text).words

    #remove stopwords
    words = [
        word for word in words if word not in set(stopwords.words('english'))
    ]

    #tag the words based on part of speech, and keep only the nouns,
    #plural nouns, verbs, and adverbs
    tagged = nltk.pos_tag(words)
    words = [
        word for (word, tag) in tagged
        if tag in ('NN', 'NNS', 'VB', 'VBZ', 'VBP', 'RB')
    ]

    #stem words into their base
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in words]
    return stemmed

