"""
This is the 'entity_naming' module.

Named entities help us understand more about what is being
referred to in a given text so that we can further classify the data.
Since named entities comprise more than one word,
it is sometimes difficult to find these from the text.

"""

# importing libraries
from pprint import pprint
import nltk
import spacy

def preprocess(sent):
    """
    >>> preprocess('European authorities fined Google on Wednesday')
    List of words and their parts of speech tag:
    [('European', 'JJ'),
     ('authorities', 'NNS'),
     ('fined', 'VBD'),
     ('Google', 'NNP'),
     ('on', 'IN'),
     ('Wednesday', 'NNP')]
     """
    resent = nltk.word_tokenize(sent)
    resent = nltk.pos_tag(resent)
    print("\nList of words and their parts of speech tag:")
    pprint(resent)

def entity_naming(sentence):
    """
    >>> entity_naming('European authorities fined Google on Wednesday')

    List of words and their entities:
    [('European', 'NORP'), ('Google', 'ORG'), ('Wednesday', 'DATE')]

    Explanation of Entities:
    [('NORP', 'Nationalities or religious or political groups'),
     ('', None),
     ('', None),
     ('ORG', 'Companies, agencies, institutions, etc.'),
     ('', None),
     ('DATE', 'Absolute or relative dates or periods')]
     """
    # loading the language that would be recognized
    nlp = spacy.load("en_core_web_sm")
    document = nlp(sentence)
    #prints the recognized words and their types
    print("\nList of words and their entities:")
    pprint([(word.text, word.label_) for word in document.ents])
    #prints the type of the word recognized and the meaning
    print("\nExplanation of Entities:")
    pprint([(word.ent_type_, spacy.explain(word.ent_type_)) for word in document])

if __name__ == '__main__':
    CORPUS = input("Enter the sentence: ")
    preprocess(CORPUS)
    entity_naming(CORPUS)
