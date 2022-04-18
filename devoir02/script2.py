# Amina DJARFI MELOUAH

import spacy
import os
import numpy as np
from nltk import FreqDist


def words_frequency(words):
    return FreqDist([w.strip() for w in words])

def load_corpus(path):
  """chargement du corpus
  """
  with open(path,'r', encoding='utf8') as fsrc:
    return fsrc.read().split('\n')


#Méthode pour avoir une liste de phrase et chaque phrase une liste de mots:
def get_sentences(docs):
  sents = []
  for doc in docs:   # en entrée c'est une liste de documents Spacy
    
    for s in doc.sents:       # pour chaque ph dans le document
      sent = [token.text.lower() for token in s if token.is_alpha] #vérifier si le token est un mot
      if len(sent)>1:       # si la liste n'est pas vide (sent représente une phrase)
        sents.append(sent)  # 
  return sents


def cell(word,shape):

  return ''


def init():
    global nlp
    nlp = spacy.load('fr_dep_news_trf')


if __name__ == "__main__":

  init()

  i=0

  file_path = '/home/amina/workspace/github/Outils_Traitements_Corpus/devoir02/data/Le_Ventre_de_Paris.txt'
  #file_path = '/home/amina/workspace/github/Outils_Traitements_Corpus/devoir02/data/test.txt'

  file_out_path = '/home/amina/workspace/github/Outils_Traitements_Corpus/devoir02/data/output-2.txt'
  
  corpus = load_corpus(file_path)

 

#POur traiter tout le fichier texte "Le_Ventre_de_Paris.txt",(corpus volumineux): 
  #https://spacy.io/usage/processing-pipelines#processing
  docs = nlp.pipe(corpus, disable=['tagger', 'ner', 'textcat','tok2vec','lemmatizer'])

  lst_phrases = get_sentences(docs)



  

  words = []
  for s in lst_phrases: # pour avoir tout le vocabulaire
    words += s  


  fdist = words_frequency(words).most_common(10)




  print("output file: '{0}'".format(file_out_path))



  with open(file_out_path , 'w') as fout:


    fout.write("chemin du fichier : '{0}'\n".format( file_path))
    fout.write("------------------------------------------------------------------------\n")

    fout.write('- nombre de mots : {0}\n'.format( len(words)))

    fout.write('|'.ljust(10,'-') + '|'.ljust(30,'-') +'|'.ljust(15,'-') +'|\n')

    fout.write('| num'.ljust(10,' ') + '| mots'.ljust(30,' ') +'| Fréquence'.ljust(15,' ') +'|\n')

    fout.write('|'.ljust(10,'-') + '|'.ljust(30,'-') +'|'.ljust(15,'-') +'|\n')
    i=0
    for k,v in dict(fdist).items():
      i+=1
      fout.write(f'| {i} '.ljust(10,' ') + f'| {k}'.ljust(30,' ') +f'| {v}'.ljust(15,' ') +'|\n')
      
    
    fout.write('|'.ljust(10,'-') + '|'.ljust(30,'-') +'|'.ljust(15,'-') +'|\n')




  
