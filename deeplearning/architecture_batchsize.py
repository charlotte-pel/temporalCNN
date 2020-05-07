#!/usr/bin/python

""" 
	Defining keras architecture.
	4.6. What values are used for the batch size?
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
def Archi_3CONV64_1FC256(X, nbclasses):
	
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
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256')	
	

#-----------------------------------------------------------------------		
#-----------------------------------------------------------------------
#--------------------- Switcher for running the architectures
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	
	
	switcher = {
		0: Archi_3CONV64_1FC256,		
		1: Archi_3CONV64_1FC256,		
		2: Archi_3CONV64_1FC256,		
		3: Archi_3CONV64_1FC256,		
		4: Archi_3CONV64_1FC256
	}
	func = switcher.get(noarchi, lambda: 0)
	if noarchi==0:
		batch_size = 8
	elif noarchi==1:
		batch_size = 16
	elif noarchi==2:
		batch_size = 32
	elif noarchi==3:
		batch_size = 64
	elif noarchi==4:
		batch_size = 128
	print("batch_size: ", batch_size)
	model = func(args[0], args[1].shape[1])
		
	if len(args)==5:
		return trainTestModel(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel(model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
