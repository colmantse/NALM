# coding=utf-8
# Copyright 2022 The NALM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ptb2016 processing code for ngram based permutation dataset.

This script collect ngrams from the train_path. The data from train_path, 
valid_path and test_path will be subsequently shuffled and displaced based
on the collected ngrams.

See "Assessing Non-autoregressive Alignment in Neural Machine Translation via
Word Reordering" for the full description of the data processing.
"""

import random, copy, string
import pickle, os

prune_threshold = 2 # the higher the threshold, the less word order retained
ngram_set_path = './ptbngrambank/ptb2016ngramsets'+'_'+str(prune_threshold)
ratios = [0.4,0.6,0.8]
flip_prob = [0.5]
train_path='./tmp/t2t_datagen/ptb2016/ptb2016.train.txt'
valid_path='./tmp/t2t_datagen/ptb2016/ptb2016.valid.txt'
test_path ='./tmp/t2t_datagen/ptb2016/ptb2016.test.txt'

destination_path='./tmp/t2t_datagen/ptb2016/'

def load_data(path):
  with open(path) as f:
    text = f.read()
  sents = text.split('\n')
  sents = [sent.split(' ')[:-1] for sent in sents]
  return sents

if os.path.exists(ngram_set_path):
  with open(ngram_set_path,'rb') as f:
    ngramsets = pickle.load(f)
else:
  import nltk
  sents = load_data(train_path)
  #below code adapted from https://www.geeksforgeeks.org/n-gram-language-modelling-with-nltk/
  stop_words = set(nltk.corpus.stopwords.words('english'))
  string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—'
  removal_list = list(stop_words) + list(string.punctuation)+ ['lt','rt']
  bigram,trigram,quadgram=[],[],[]
  for sentence in sents:
    sentence = list(map(lambda x:x.lower(),sentence))
    for word in sentence:
      if word== '.':
        sentence.remove(word)
    bigram.extend(list(nltk.ngrams(sentence, 2)))
    trigram.extend(list(nltk.ngrams(sentence, 3)))
    quadgram.extend(list(nltk.ngrams(sentence, 4)))
  def remove_stopwords(x):    
    y = []
    for pair in x:
      count = 0
      for word in pair:
        if word in removal_list:
          count = count or 0
        else:
          count = count or 1
      if (count==1):
        y.append(pair)
    return (y)
  bigram=nltk.FreqDist(remove_stopwords(bigram))
  trigram=nltk.FreqDist(remove_stopwords(trigram))
  quadgram=nltk.FreqDist(remove_stopwords(quadgram))
  #end adaptation
  ngramsets={}
  ngramsets['bi']=set([x for x in bigram if bigram[x]>prune_threshold])
  ngramsets['tri']=set([x for x in trigram if trigram[x]>prune_threshold])
  ngramsets['quad']=set([x for x in quadgram if quadgram[x]>prune_threshold])
  with open(ngram_set_path,'wb') as f:
    pickle.dump(ngramsets,f)

def find_gram(sent):
  for i in range(len(sent)):
    ngram = tuple(map(lambda x:x.lower(),sent[i:i+4]))
    if ngram in ngramsets['quad']:
      return True, i, i+4
  for i in range(len(sent)):
    ngram = tuple(map(lambda x:x.lower(),sent[i:i+3]))
    if ngram in ngramsets['tri']:
      return True, i, i+3
  for i in range(len(sent)):
    ngram = tuple(map(lambda x:x.lower(),sent[i:i+2]))
    if ngram in ngramsets['bi']:
      return True, i, i+2
  return False, 0, 0

def ngramize_body(sent):
  continue_gram,i,j = find_gram(sent)
  if continue_gram:
    return ngramize_body(sent[:i]),[' '.join(sent[i:j])],ngramize_body(sent[j:])
  return sent

def traverse_tree(node):
  if isinstance(node,tuple):
    tmp=[]
    for n in node:
      if isinstance(n,list):
        tmp.extend(n)
      else:
        tmp.extend(traverse_tree(n))
    return tmp
  else:
    return node

def ngramize(sent):
  sent_tree = ngramize_body(sent)
  return traverse_tree(sent_tree)

def isngram(token):
  return ' ' in token

def shuffle_ngram(sent, ratio):
  # this guarantee displacement of ratio *len(sent) tokens in sent
  # if ratio*len(sent) <= 1, return completely shuffled sent
  # return error if len(sent) == 1 
  if len(sent)==1:
    tmp=sent[0].split(' ')
    random.shuffle(tmp)
    return tmp
  if round(len(sent)*ratio)<=1:
    random.shuffle(sent)
    return sent
  shuff_id = random.sample([i for i in range(len(sent))],round(len(sent)*ratio))
  ngram_id = sorted(shuff_id)
  while any(ngram_id[i] == shuff_id[i] for i in range(len(ngram_id))):
    random.shuffle(ngram_id)
  tmp=copy.deepcopy(sent)
  for i,j in zip(ngram_id,shuff_id):
    tmp[i]=sent[j]
  return tmp

def shuffle_text_by_ngram(sents,ratio):
  return [shuffle_ngram(ngramize(sent),ratio) for sent in sents]

def generate_shuffle_text_by_ngram(path,sents,ratio):
  if os.path.exists(path):
    print('skipping '+path + ': already exists')
  else:
    text = shuffle_text_by_ngram(sents,ratio)
    for i, sent in enumerate(text):
      text[i]=' '.join(sent)
    with open(path,'w') as f:
      f.write('\n'.join(text))

def displace_ngram(sent, prob):
  #prob: probability that a displacement occurs
  def find_init(seg):
    return random.randint(0,len(seg)-2)
  
  def flip(biconstituent):
    # biconstituent is a list of 2 item
    assert len(biconstituent)==2
    if random.random() > prob:
      biconstituent.reverse()
    return ' '.join(biconstituent)

  while len(sent)>1:
    if len(sent)==2:
      sent = [flip(sent)]
      continue
    i = find_init(sent)
    sent=sent[:i]+[flip(sent[i:i+2])]+sent[i+2:]
  return sent[0] if len(sent)==1 else ''

def displace_text_by_ngram(sents, prob):
  return [displace_ngram(ngramize(sent),prob) for sent in sents]

def generate_adjacency_displaced_text(path,sents,prob):
  if os.path.exists(path):
    print('skipping '+path + ': already exists')
  else:
    text=displace_text_by_ngram(sents,prob)
    with open(path,'w') as f:
      f.write('\n'.join(text))

def main():
  train=load_data(train_path)
  valid=load_data(valid_path)
  test=load_data(test_path)
  for r in ratios:
    generate_shuffle_text_by_ngram(destination_path+'ptb2016.train.'+'shuffle.'+str(r)+'_p'+str(prune_threshold)+'.txt',train,r)
    generate_shuffle_text_by_ngram(destination_path+'ptb2016.valid.'+'shuffle.'+str(r)+'_p'+str(prune_threshold)+'.txt',valid,r)
    generate_shuffle_text_by_ngram(destination_path+'ptb2016.test.'+'shuffle.'+str(r)+'_p'+str(prune_threshold)+'.txt',test,r)
  for p in flip_prob:
    generate_adjacency_displaced_text(destination_path+'ptb2016.train.'+'displace.'+str(p)+'_p'+str(prune_threshold)+'.txt',train,p)
    generate_adjacency_displaced_text(destination_path+'ptb2016.valid.'+'displace.'+str(p)+'_p'+str(prune_threshold)+'.txt',valid,p)
    generate_adjacency_displaced_text(destination_path+'ptb2016.test.'+'displace.'+str(p)+'_p'+str(prune_threshold)+'.txt',test,p)

if __name__ == "__main__":
  main()

