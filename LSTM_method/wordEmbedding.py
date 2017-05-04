# coding = utf-8
import six.moves.cPickle as pickle
from itertools import chain
from collections import Counter  # Count up the words occurrence times
#import matplotlib.pyplot as plt  # For test
import os
import numpy as np 


# Constant values

vob_size = 2800  # Sample data number
seed = 12345  # Random seed
# embed_dim = 100
# is_lower = False
dim = 100

with open('data/tokens.pkl', 'rb') as f:
	heads, contents, keywords = pickle.load(f)



def get_vocab(words_list):
	# Count is a dict
	count = Counter(c for text in words_list for c in text.split())
	# Sort desc
	vocab = map(lambda x: x[0], sorted(count.items(), key = lambda x:-x[1]))
	return vocab, count


# Here vocab is iterator
vocab, count = get_vocab(heads + contents)
vocab = list(vocab)


##################################################
# Index words

empty = 0
eos = 1
start_id = eos+1

def get_id(vocab, count):
	word_to_id = dict((word, id+start_id) for id, word in enumerate(vocab))
	word_to_id['<empty>'] = empty
	word_to_id['<eos>'] = eos
	id_to_word = dict((id, word) for word, id in word_to_id.items())
	return word_to_id, id_to_word

word_to_id, id_to_word = get_id(vocab, count)


###################################################
file_name = 'glove.6B.100d.txt'
data_dir = '/home/angus/work/mastercourse/544/project/headlines-master/data/'

glove_name = os.path.join(data_dir, file_name)

# Glove.6B.100d.txt has 400k data in total
glove_data_number = 400000

glove_index_dict={}
glove_embedding_weights = np.empty((glove_data_number, 100))
glove_scale = 0.1

with open(glove_name, 'r') as f:
	i = 0
	for line in f:
		line = line.strip().split()
		word = line[0]

		glove_index_dict[word] = i
		glove_embedding_weights[i,:] = list(map(float,line[1:]))
		i+=1
glove_embedding_weights = glove_embedding_weights * glove_scale


for word, i in glove_index_dict.items():
	word = word.lower()
	if word not in glove_index_dict:
		glove_index_dict[word] = i



######################################################
# np.random.seed(seed)

shape = (vob_size, dim)
scale = glove_embedding_weights.std() * np.sqrt(12)/2
embedding = np.random.uniform(low=-scale, high=scale, size=shape)

c = 0
for i in range(vob_size):
	word = id_to_word[i]
	g = glove_index_dict.get(word, glove_index_dict.get(word.lower()))

	if g is None and word.startswith('#'):
		word = word[1:]
		g = glove_index_dict.get(word, glove_index_dict.get(word.lower()))

	if g is not None:
		embedding[i,:] = glove_embedding_weights[g,:]
		c+=1



threadhold = 0.5
word_to_glove={}

for word in word_to_id:
	if word in glove_index_dict:
		g = word
	elif word.lower() in glove_index_dict:
		g = word.lower()
	elif word.startswith('#') and word[1:] in glove_index_dict:
		g = word[1:]
	elif word.startswith('#') and word[1:].lower() in glove_index_dict:
		g = word[1:].lower()
	else:
		continue
	word_to_glove[word] = g



normed_embedding = embedding / np.array([np.sqrt(np.dot(w,w)) for w in embedding])[:,None]


unknown_words = 100

glove_match=[]

for word, index in word_to_id.items():
	if index >= vob_size - unknown_words and word.isalpha() and word in word_to_glove:
		g_index = glove_index_dict[word_to_glove[word]]
		g_weight = glove_embedding_weights[g_index,:].copy()
		g_weight = g_weight / np.sqrt(np.dot(g_weight,g_weight))
		score = np.dot(normed_embedding[:vob_size - unknown_words], g_weight)

		while True:
			embedding_index = score.argmax()
			s =score[embedding_index]
			if s < threadhold:
				break
			if id_to_word[embedding_index] in word_to_glove:
				glove_match.append((word, embedding_index, s))
				break
			score[embedding_index] = -1

# Sort desc
glove_match.sort(key = lambda x:-x[2])

# Lookup table
glove_idx_index = dict((word_to_id[w],embedding_index) for  w, embedding_index, _ in glove_match)




with open('data/wordEmbeddings.pkl', 'wb') as f:
	pickle.dump((embedding, id_to_word, word_to_id, glove_idx_index), f, -1)


X = [[word_to_id[token] for token in d.split()] for d in contents]
Y = [[word_to_id[token] for token in headline.split()] for headline in heads]

with open('data/wordEmbeddings.data.pkl', 'wb') as f:
	pickle.dump((X,Y), f, -1)
