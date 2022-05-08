"""


Question 2 : Compter le nombre de phrases où apparaissent les mots “travail” et “manger” (sur la base du lemme).


"""



import nltk
#nltk.download('punkt')
import pprint as pp
import spacy
from nltk import word_tokenize



def tokenizer(text):
    """onvertir text en une liste de mots/tokens     
    """

    #POur traiter tout le fichier texte "Le_Ventre_de_Paris.txt",(corpus volumineux): 
  #https://spacy.io/usage/processing-pipelines#processing
    docs = nlp.pipe(text, disable=['tagger', 'ner', 'textcat','tok2vec'])

    tokens = []
    
    for doc in docs:

        tokens+=[(token.text, token.lemma_, token.pos_) for token in doc]
        
        
    return tokens


def collocations_bigrams(tokens):
    """
    """
    return  nltk.collocations.BigramAssocMeasures(), nltk.collocations.BigramCollocationFinder.from_words(tokens)



def collocations_trigrams(tokens):
    """
    """
    return  nltk.collocations.TrigramAssocMeasures(), nltk.collocations.TrigramCollocationFinder.from_words(tokens)



def load_corpus(path_file):
    """
     chargement du corpus
    """
    with open(path_file, "r", encoding = "UTF-8") as fsrc:
        return fsrc.read().split('\n')


def init():
    global nlp
    nlp = spacy.load('fr_dep_news_trf')




def ngrams_filter(ngrams, cond):
    """
    filtre (supprime) les ngrams non inclus dans la liste 

    return:
    True : ngrams => supprimé
    False : ngrams => selectionné

    """   
    return not (any(w for w in ngrams if w in cond))




init()

# chargement du corpus
text = load_corpus("Le_Ventre_de_Paris.txt")

lst_tokens = tokenizer(text[:500])


bigrams, bigramFinder = collocations_bigrams(lst_tokens)


condition = ["travail", "manger"]


bigramFinder.apply_ngram_filter(lambda w1, w2: ngrams_filter((w1[1],w2[1]) ,condition))

print("nombre bigrams:", len(bigramFinder.ngram_fd.items()))


trigrams, trigramFinder = collocations_trigrams(lst_tokens)

trigramFinder.apply_ngram_filter(lambda w1, w2, w3: ngrams_filter((w1[1],w2[1],w3[1]) ,condition))

print("nombre trigrams:",len(trigramFinder.ngram_fd.items()))



