"""
Question 1 : Explorer des collocations avec Spacy (ou NLTK)

"""


import nltk
nltk.download('punkt')
import pprint as pp

from nltk import word_tokenize



def tokenizer(text,language = "french"):
    """onvertir text en une liste de mots/tokens     
    """
    return word_tokenize(text, language = language)



def collocations_bigrams(tokens):
    """
    """
    return  nltk.collocations.BigramAssocMeasures(), nltk.collocations.BigramCollocationFinder.from_words(tokens)



def collocations_trigrams(tokens):
    """
    """
    return  nltk.collocations.TrigramAssocMeasures(), nltk.collocations.TrigramCollocationFinder.from_words(tokens)



def load_corpus(path):
  """chargement du corpus
  """
  with open(path,'r', encoding='utf8') as fsrc:
    return fsrc.read()






# chargement du corpus
text = load_corpus("Le_Ventre_de_Paris.txt")

lst_tokens = tokenizer(text)

bigrams, bigramFinder = collocations_bigrams(lst_tokens)

print("liste bigrams:")
pp.pprint(bigramFinder.nbest(bigrams.pmi, 20))


trigrams, trigramFinder = collocations_trigrams(lst_tokens)

print("liste trigrams:")
pp.pprint(trigramFinder.nbest(trigrams.pmi, 20))