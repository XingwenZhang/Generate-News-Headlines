# coding = utf-8


import os
import keras
import six.moves.cPickle as pickle 
from sklearn.model_selection import train_test_split
import numpy as np 
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K 
from keras.optimizers import Adam, RMSprop
import random
import sys
#import Levenshtein  # Edit distance


content_max_len = 25
head_max_len = 25
max_len = content_max_len + head_max_len

rnn_size = 512
rnn_layers = 3
batch_norm = False
seed = 42
activation_rnn_size = 40 if head_max_len else 0



# Training parameters
optimizer = 'adam'
LR = 1e-4
batch_size = 64
nflips = 10
dropout = 0
recurrent_dropout = 0
weight_decay = 0
model_dropout = 0
#p_W, p_U, p_dense, weight_decay = 0, 0, 0, 1

train_samples = 2000
validation_samples = 800

with open('data/wordEmbeddings.pkl', 'rb') as f:
	embedding, index_to_word, word_to_index, glove_index_to_index = pickle.load(f)

vob_size, embedding_size = embedding.shape

with open('data/wordEmbeddings.data.pkl', 'rb') as f:
	X, Y = pickle.load(f)

unknown_words = 10

for i in range(unknown_words):
	index_to_word[vob_size-1-i] = '<%d>'%i

known_words = vob_size - unknown_words

for i in range(known_words, len(index_to_word)):
	index_to_word[i] = index_to_word[i] + '^'

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_samples, random_state=seed)

del X
del Y


empty = 0
eos = 1
index_to_word[empty] = '_'
index_to_word[eos] = '~'



#############################
# Model train

regularizer = l2(weight_decay) if weight_decay else None

model = Sequential()
model.add(Embedding(vob_size, embedding_size,
	input_length=max_len,
	embeddings_regularizer=regularizer,
	weights=[embedding],
	mask_zero=True,
	name='embedding_1'))

for i in range(rnn_layers):
	lstm = LSTM(rnn_size, return_sequences=True,
		kernel_regularizer=regularizer,
		recurrent_regularizer=regularizer,
		bias_regularizer=regularizer,
		dropout=dropout,
		recurrent_dropout=recurrent_dropout,
		name='lstm_%d'%(i+1))
	model.add(lstm)
	model.add(Dropout(model_dropout, name='dropout_%d'%(i+1)))



def simple_context(X, mask, n=activation_rnn_size, content_max_len=content_max_len, head_max_len=head_max_len):
	content, head = X[:,:content_max_len,:], X[:,content_max_len:,:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	content_activations, content_words = content[:,:,:n], content[:,:,n:]

	activation_energies = K.batch_dot(head_activations, content_activations,axes=(2,2))

	activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :content_max_len],'float32'),1)

	# for every head word compute weights for every contents word
	activation_energies = K.reshape(activation_energies,(-1,content_max_len))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,head_max_len,content_max_len))

    # for every head word compute weighted average of content words
	content_avg_word = K.batch_dot(activation_weights, content_words, axes=(2,1))
	return K.concatenate((content_avg_word, head_words))


if activation_rnn_size:
	model.add(Lambda(simple_context,
		mask = lambda inputs, mask: mask[:,content_max_len:],
		output_shape = lambda input_shape: (input_shape[0], head_max_len, 2*(rnn_size - activation_rnn_size)),
		name='simplecontext_1'))
model.add(TimeDistributed(Dense(vob_size,
                                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                name = 'timedistributed_1')))
model.add(Activation('softmax', name='activation_1'))


model.compile(loss='categorical_crossentropy', optimizer=optimizer)

K.set_value(model.optimizer.lr,np.float32(LR))
model.summary()
print("here")


def lpadd(x, content_max_len=content_max_len, eos=eos):
	"""left (pre) pad a description to maxlend and then add eos.
	The eos is the input to predicting the first word in the headline
	"""
	assert content_max_len >= 0
	if content_max_len == 0:
		return [eos]
	n = len(x)
	if n > content_max_len:
		x = x[-content_max_len:]
		n = content_max_len
	return [empty]*(content_max_len-n) + x + [eos]


samples = [lpadd([3]*26)]
data = sequence.pad_sequences(samples, maxlen=max_len, value=empty, padding='post', truncating='post')


np.all(data[:,content_max_len] == eos)
data.shape,map(len, samples)
probs = model.predict(data, verbose=0, batch_size=1)
probs.shape


def flip_headline(x, nflips=None, model=None, debug=False):
	"""given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
	with words predicted by the model
	"""
	if nflips is None or model is None or nflips <= 0:
		return x
    
	batch_size = len(x)
	assert np.all(x[:,content_max_len] == eos)
	probs = model.predict(x, verbose=0, batch_size=batch_size)
	x_out = x.copy()
	for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
		flips = sorted(random.sample(range(content_max_len+1,max_len), nflips))
		if debug and b < debug:
			print (b,)
		for input_idx in flips:
			if x[b,input_idx] == empty or x[b,input_idx] == eos:
				continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
			label_idx = input_idx - (content_max_len+1)
			prob = probs[b, label_idx]
			w = prob.argmax()
			if w == empty:  # replace accidental empty with oov
				w = known_words
			if debug and b < debug:
				print ('%s => %s'%(index_to_word[x_out[b,input_idx]],index_to_word[w]),)
			x_out[b,input_idx] = w
		if debug and b < debug:
			print
	return x_out



def vocab_fold(xs):
	"""convert list of word indexes that may contain words outside vob_size to words inside.
	If a word is outside, try first to use glove_index_to_index to find a similar word inside.
	If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
	"""
	xs = [x if x < known_words else glove_index_to_index.get(x,x) for x in xs]
	# the more popular word is <0> and so on
	outside = sorted([x for x in xs if x >= known_words])
	# if there are more than unknown_words oov words then put them all in unknown_words-1
	outside = dict((x,vob_size-1-min(i, unknown_words-1)) for i, x in enumerate(outside))
	xs = [outside.get(x,x) for x in xs]
	return xs


def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
	"""description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
	batch_size = len(xhs)
	assert len(xds) == batch_size
	x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
	x = sequence.pad_sequences(x, maxlen=max_len, value=empty, padding='post', truncating='post')
	x = flip_headline(x, nflips=nflips, model=model, debug=debug)
    
	y = np.zeros((batch_size, head_max_len, vob_size))
	for i, xh in enumerate(xhs):
		xh = vocab_fold(xh) + [eos] + [empty]*head_max_len  # output does have a eos at end
		xh = xh[:head_max_len]
		y[i,:,:] = np_utils.to_categorical(xh, vob_size)
        
	return x, y


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
	"""yield batches. for training use nb_batches=None
	for validation generate deterministic results repeating every nb_batches
	
	while training it is good idea to flip once in a while the values of the headlines from the
	value taken from Xh to value generated by the model.
	"""
	c = nb_batches if nb_batches else 0
	while True:
		xds = []
		xhs = []
		if nb_batches and c >= nb_batches:
			c = 0
		new_seed = random.randint(0, sys.maxsize)
		random.seed(c+123456789+seed)
		for b in range(batch_size):
			t = random.randint(0,len(Xd)-1)

			xd = Xd[t]
			s = random.randint(min(content_max_len,len(xd)), max(content_max_len,len(xd)))
			xds.append(xd[:s])
            
			xh = Xh[t]
			s = random.randint(min(head_max_len,len(xh)), max(head_max_len,len(xh)))
			xhs.append(xh[:s])
		# undo the seeding before we yield inorder not to affect the caller
		c+= 1
		random.seed(new_seed)
		yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)






##################################
# Training Model

history = {}

train_gen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
val_gen = gen(X_test, Y_test, nb_batches=validation_samples//batch_size, batch_size=batch_size)


for iteration in range(10):
	print ('Iteration', iteration)
	h = model.fit_generator(train_gen, steps_per_epoch=train_samples//batch_size,
		epochs=1, validation_data=val_gen, validation_steps=validation_samples)
	for k,v in h.history.items():
		history[k] = history.get(k,[]) + v
	with open('data/train.history.pkl','wb') as fp:
		pickle.dump(history,fp,-1)
	model.save_weights('data/train.hdf5', overwrite=True)
	#gensamples(batch_size=batch_size)

