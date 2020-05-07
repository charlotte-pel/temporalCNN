#!/usr/bin/python

""" 
	Defining keras architecture.
	4.5. How to control overfitting?
"""


import sys, os

from deeplearning.architecture_features import *
import keras
from keras import layers
from keras.layers import Flatten
from keras import backend as K




#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#------------------------------------------------------------------------	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_onlyDropout(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 0
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_onlyDropout')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_onlyBN(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 0
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate))
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_onlyBN')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_onlyWD(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_relu(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate))
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_relu(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_onlyWD')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_onlyVal(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 0
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_relu(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate))
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_relu(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_onlyVal')	
	

#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_withoutDropout(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate))
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_withoutDropout')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_withoutBN(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_withoutBN')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_withoutWD(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 0.
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_withoutWD')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_withoutVal(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_withoutVal')	
	

#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_nothing(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 0.
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 64 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_relu(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate))
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_relu(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate))
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_nothing')	
	
#-----------------------------------------------------------------------		
#-----------------------------------------------------------------------
#--------------------- Switcher for running the architectures
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	batch_size = 32
	
	switcher = {
		0: Archi_3CONV64_1FC256_onlyDropout,
		1: Archi_3CONV64_1FC256_onlyBN,
		2: Archi_3CONV64_1FC256_onlyWD,
		3: Archi_3CONV64_1FC256_onlyVal,
		
		4: Archi_3CONV64_1FC256_withoutDropout,
		5: Archi_3CONV64_1FC256_withoutBN,
		6: Archi_3CONV64_1FC256_withoutWD,
		7: Archi_3CONV64_1FC256_withoutVal,
		
		8: Archi_3CONV64_1FC256_nothing,
		
					
	}
	func = switcher.get(noarchi, lambda: 0)
	model = func(args[0], args[1].shape[1])
	
	if len(args)==5:
		return trainTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
