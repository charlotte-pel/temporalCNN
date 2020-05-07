#!/usr/bin/python

""" 
	Defining keras architecture.
	4.1. Benefiting from both spectral and temporal dimensions.
	4.2. Influence of the filter size
"""

import sys, os

from deeplearning.architecture_features import *
import keras
from keras import layers
from keras.layers import Flatten
from keras import backend as K

#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#-----------------------------------------------------------------------
	
#------------------------------------------------------------------------	
def Archi_3FC1024(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nbunits_fc = 1024
	nb_fc = 3
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- Flatten + 	3 FC layers
	X = Flatten()(X_input)	
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3FC1024')


#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_f3f1(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth, _ = X.shape
	input_shape = (L,depth,1)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = [192, 64, 21] #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv2d_bn_relu_drop(X, nbunits=nbunits_conv[add], kernel_size=(3,1), kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_f3f1')
		
			
#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_f5f1(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth, _ = X.shape
	input_shape = (L,depth,1)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = [192, 64, 21] #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv2d_bn_relu_drop(X, nbunits=nbunits_conv[add], kernel_size=(5,1), kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_f5f1')	
		
		
#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_f9f1(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth, _ = X.shape
	input_shape = (L,depth,1)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = [192, 64, 21] #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv2d_bn_relu_drop(X, nbunits=nbunits_conv[add], kernel_size=(9,1), kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_f9f1')
	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_f17f1(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth, _ = X.shape
	input_shape = (L,depth,1)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = [192, 64, 21] #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv2d_bn_relu_drop(X, nbunits=nbunits_conv[add], kernel_size=(17,1), kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_f17f1')
	
	
#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_f33f1(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth, _ = X.shape
	input_shape = (L,depth,1)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = [192, 64, 21] #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv2d_bn_relu_drop(X, nbunits=nbunits_conv[add], kernel_size=(33,1), kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_f33f1')		


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_f1fd(X, nbclasses):
	
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
	X = conv_bn_relu_drop(X_input, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_f1fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_f3fd(X, nbclasses):
	
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
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=3, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_f3fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_f5fd(X, nbclasses):
	
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
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_f5fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_f9fd(X, nbclasses):
	
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
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=9, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_f9fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_f17fd(X, nbclasses):
	
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
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=17, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_f17fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64_1FC256_f33fd(X, nbclasses):
	
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
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=33, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256_f33fd')	


#-----------------------------------------------------------------------		
#-----------------------------------------------------------------------
#--------------------- Switcher for running the architectures
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	batch_size = 32
	
	switcher = {
		0: Archi_3FC1024,
		
		10: Archi_3CONV64C_1FC256_f3f1,			
		11: Archi_3CONV64C_1FC256_f5f1,			
		12: Archi_3CONV64C_1FC256_f9f1,			
		13: Archi_3CONV64C_1FC256_f17f1,			
		14: Archi_3CONV64C_1FC256_f33f1,		
		
		1: Archi_3CONV64_1FC256_f1fd,
		
		20: Archi_3CONV64_1FC256_f3fd,
		21: Archi_3CONV64_1FC256_f5fd,
		22: Archi_3CONV64_1FC256_f9fd,
		23: Archi_3CONV64_1FC256_f17fd,
		24: Archi_3CONV64_1FC256_f33fd					
	}
	func = switcher.get(noarchi, lambda: 0)
	model = func(args[0], args[1].shape[1])
	
	if len(args)==5:
		return trainTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
