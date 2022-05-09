"""


Question 4 : Bonus : un nuage de mots avec from wordcloud import WordCloud (ou un autre outil).



"""

import wordcloud as wc
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import pprint as pp

from nltk import word_tokenize
from nltk.corpus import stopwords
import string 

def tokenizer(text,language = "french"):
    """onvertir text en une liste de mots/tokens     
    """
    lst = word_tokenize(text, language = language)
    return [w.lower() for w in lst if w not in string.punctuation]



def collocations_ngrams(tokens):
    """
    """
    return  nltk.collocations.BigramAssocMeasures, nltk.collocations.BigramCollocationFinder.from_words(tokens)



def load_corpus(path_file):
    """
     chargement du corpus
    """
    with open(path_file, "r", encoding = "UTF-8") as fsrc:
        return fsrc.read()





# chargement du corpus
text = load_corpus("Le_Ventre_de_Paris.txt")

lst_tokens = tokenizer(text)

print('--------------------------------------------')

ngrams, ngramFinder = collocations_ngrams(lst_tokens)


words = []

for ng in ngramFinder.nbest(ngrams.pmi, 1000):
    words += [w for w in ng]

print(words[:10])
#TODO CONTEXT




wordcloudimage = WordCloud( max_words=250,
                            font_step=2 ,
                            max_font_size=250,
                            stopwords=stopwords.words('french'),
                            background_color='black',
                            width=1000,
                            height=720
                          ).generate(' '.join(words))
 
plt.figure(figsize=(20,8))
plt.imshow(wordcloudimage)
plt.axis("off")
plt.show()


wordcloudimage.to_file('wordcloudimage.png')