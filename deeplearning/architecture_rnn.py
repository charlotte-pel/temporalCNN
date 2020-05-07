#!/usr/bin/python

""" 
	Defining keras architecture.
	4.1 Benefiting from both spectral and temporal dimensions.
"""

import sys, os

from deeplearning.architecture_features import *
import keras
from keras import layers
from keras.layers import GRU, CuDNNGRU, Bidirectional
from keras import backend as K


#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#------------------------------------------------------------------------	

#-----------------------------------------------------------------------		
def Archi_3GRU16BI_1FC256(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_rnn = 3
	nbunits_rnn = 16
	nbunits_fc = 256
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_rnn-1):
		X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=True))(X)
		X = Dropout(dropout_rate)(X)
	X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=False))(X)
	X = Dropout(dropout_rate)(X)
	#-- 1 FC layers
	X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3GRU16BI_1FC256')
	
#-----------------------------------------------------------------------		
def Archi_3GRU32BI_1FC256(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	#~ dropout_rate = 0.5
	nb_rnn = 3
	nbunits_rnn = 32
	nbunits_fc = 256
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_rnn-1):
		X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=True))(X)
		X = Dropout(dropout_rate)(X)
	X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=False))(X)
	X = Dropout(dropout_rate)(X)
	#-- 1 FC layers
	X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3GRU32BI_1FC256')

#-----------------------------------------------------------------------		
def Archi_3GRU64BI_1FC256(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_rnn = 3
	nbunits_rnn = 64
	nbunits_fc = 256
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_rnn-1):
		X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=True))(X)
		X = Dropout(dropout_rate)(X)
	X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=False))(X)
	X = Dropout(dropout_rate)(X)
	#-- 1 FC layers
	X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3GRU64BI_1FC256')

	
#-----------------------------------------------------------------------		
def Archi_3GRU128BI_1FC256(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_rnn = 3
	nbunits_rnn = 128
	nbunits_fc = 256
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_rnn-1):
		X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=True))(X)
		X = Dropout(dropout_rate)(X)
	X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=False))(X)
	X = Dropout(dropout_rate)(X)
	#-- 1 FC layers
	X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3GRU128BI_1FC256')

		
#-----------------------------------------------------------------------		
def Archi_3GRU256BI_1FC256(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_rnn = 3
	nbunits_rnn = 256
	nbunits_fc = 256
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_rnn-1):
		X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=True))(X)
		X = Dropout(dropout_rate)(X)
	X = Bidirectional(CuDNNGRU(nbunits_rnn, return_sequences=False))(X)
	X = Dropout(dropout_rate)(X)
	#-- 1 FC layers
	X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3GRU256BI_1FC256')	

#--------------------- Switcher for running the architectures
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	batch_size = 32
	
	switcher = {
		0: Archi_3GRU16BI_1FC256,
		1: Archi_3GRU32BI_1FC256,
		2: Archi_3GRU64BI_1FC256,
		3: Archi_3GRU128BI_1FC256,
		4: Archi_3GRU256BI_1FC256
	}
	func = switcher.get(noarchi, lambda: 0)
	model = func(args[0], args[1].shape[1])
	
	if len(args)==5:
		return trainTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
