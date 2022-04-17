# Amina DJARFI Melouah

import spacy
import os
import numpy as np


def load_corpus(path):
  """chargement du corpus
  """
  with open(path,'r', encoding='utf8') as fsrc:
    return fsrc.read().split('\n')

"""""
#1.La taille du fichier en octets

file_size = os.path.getsize(r'devoir02/data/Le_Ventre_de_Paris.txt') 
print('File Size:', file_size, 'bytes')
"""

def file_size(path, unit=""):
  """calcul de la taille du fichier
  """
  dv =1
  if unit=="Ko":
    dv = 1024

  if unit=="Mo":
    dv = 1024**2

  return np.round(os.path.getsize(path) /dv,2) 

def get_sentences(docs):
  sents = []
  for doc in docs:
    
    for s in doc.sents:
      tokens = [token.text for token in s if token.is_alpha]
      if len(tokens)>1:       
        sents.append(tokens)
  return sents


def type_token_ratio(lst):
  words = []
  for s in lst:
    words += s   

  return np.round(len(set(words))/len(words) * 100,2)


def calcul_avr_sent(lst):
  return np.round(np.average( [len(s) for s in lst]),2)

def calcul_ecart_type(lst):
  return np.round(np.std( [len(s) for s in lst]),2)


def init():
    global nlp
    nlp = spacy.load('fr_dep_news_trf')


if __name__ == "__main__":

  init()

  i=0

  file_path = '/home/amina/workspace/github/Outils_Traitements_Corpus/devoir02/data/Le_Ventre_de_Paris.txt'

  file_out_path = '/home/amina/workspace/github/Outils_Traitements_Corpus/devoir02/data/output-1.txt'
  
  corpus = load_corpus(file_path)

  taille_fichier = file_size(file_path,"Mo")


  


  #https://spacy.io/usage/processing-pipelines#processing
  docs = nlp.pipe(corpus, disable=['tagger', 'ner', 'textcat','tok2vec','lemmatizer'])

  lst_phrases = get_sentences(docs)

  

  

 

  nbr_phrases = len(lst_phrases)

  long_moyen_phrases = calcul_avr_sent(lst_phrases)

  ecart_type = calcul_ecart_type(lst_phrases)

  rtt = type_token_ratio(lst_phrases)

  print("output file: '{0}'".format(file_out_path))

  with open(file_out_path , 'w') as fout:


    fout.write("chemin du fichier : '{0}'\n".format( file_path))
    fout.write("------------------------------------------------------------------------\n")
    i+=1
    fout.write('{0} - taille du fichier : {1} Mo\n'.format(i, taille_fichier))
  
    i+=1
    fout.write('{0} - ratio type/token : {1}\n'.format(i, rtt))

    i+=1
    fout.write('{0} - nombre de phrases dans le texte : {1}\n'.format(i,nbr_phrases))

    i+=1
    fout.write('{0} - longueur de phrase moyenne (en mots) : {1}\n'.format(i,long_moyen_phrases))

    i+=1
    fout.write('{0} - écart moyen de la longueur des phrases (en mots) : {1}\n'.format(i,ecart_type))
  #
 

