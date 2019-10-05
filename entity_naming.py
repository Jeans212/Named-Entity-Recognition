# importing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from pprint import pprint
import spacy
import en_core_web_sm

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    print("\nList of words and their parts of speech tag:")
    pprint(sent)

def entity_naming():
    # loading the language that would be recognized
    nlp = spacy.load("en_core_web_sm")
    document = nlp(corpus)

    #prints the recognized words and their types
    print("\nList of words and their entities:")
    pprint([(word.text, word.label_) for word in document.ents])
    
    #prints the type of the word recognized and the meaning
    print("\nExplanation of Entities:")
    pprint([(word.ent_type_, spacy.explain(word.ent_type_)) for word in document])

if __name__=='__main__':
    corpus = input("Enter the sentence: ")
    preprocess(corpus)
    entity_naming()
