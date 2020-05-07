#!/usr/bin/python

""" 
	Defining keras architecture. 
	4.3. Are local and global temporal pooling layers important?
"""


import sys, os

from deeplearning.architecture_features import *
import keras
from keras import layers
from keras.layers import AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Flatten
from keras import backend as K

	
#-----------------------------------------------------------------------
#---------------------- ARCHITECTURES
#------------------------------------------------------------------------	

#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_GAP_f3fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 640 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=3, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_GAP_f3fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_GAP_f5fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 512 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_GAP_f5fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_GAP_f9fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 384 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=9, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_GAP_f9fd')	

#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_GAP_f17fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 256 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=17, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_GAP_f17fd')	



#-----------------------------------------------------------------------		
def Archi_3CONV64C_1FC256_GAP_f33fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 192 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = X_input
	for add in range(nb_conv):
		X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=33, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV64C_1FC256_GAP_f33fd')	




#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_f33_17_9fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=128, kernel_size=33, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=192, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=256, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_f33_17_9fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_f17_9_5fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=128, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=256, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=384, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_f17_9_5fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_f9_5_3fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=nbunits_conv, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**1, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**2, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_f9_5_3fd')	

	
#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_f5_3_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**1, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**1, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_f5_3_1fd')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_f3_1_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=nbunits_conv, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_f3_1_1fd')	
	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_f33_17_9fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=128, kernel_size=33, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=192, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=256, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_f33_17_9fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_f17_9_5fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=128, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=256, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=384, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_f17_9_5fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_f9_5_3fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=nbunits_conv, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**1, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**2, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_f9_5_3fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_f5_3_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**1, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv*2**1, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	#~ X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_f5_3_1fd')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_f3_1_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=nbunits_conv, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=nbunits_conv, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = Flatten()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_f3_1_1fd')	
	

#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_GAP_f33_17_9fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=256, kernel_size=33, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=256, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_GAP_f33_17_9fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_GAP_f17_9_5fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=256, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_GAP_f17_9_5fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_GAP_f9_5_3fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=512, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_GAP_f9_5_3fd')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_GAP_f5_3_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=512, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=768, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=1024, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	#~ X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_GAP_f5_3_1fd')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2MP_1FC256_GAP_f3_1_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=768, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=1024, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=1024, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2MP_1FC256_GAP_f3_1_1fd')	

		
#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_GAP_f33_17_9fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_conv = 128 #-- will be double
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=256, kernel_size=33, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=256, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_GAP_f33_17_9fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_GAP_f17_9_5fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=256, kernel_size=17, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_GAP_f17_9_5fd')	


#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_GAP_f9_5_3fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=512, kernel_size=9, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=512, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_GAP_f9_5_3fd')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_GAP_f5_3_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=512, kernel_size=5, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=768, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=1024, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_GAP_f5_3_1fd')	
	
#-----------------------------------------------------------------------		
def Archi_3CONV2AP_1FC256_GAP_f3_1_1fd(X, nbclasses):
	
	#-- get the input sizes
	m, L, depth = X.shape
	input_shape = (L,depth)
	
	#-- parameters of the architecture
	l2_rate = 1.e-6
	dropout_rate = 0.5
	nb_conv = 3
	nb_fc= 1
	nbunits_fc = 256 #-- will be double
	
	# Define the input placeholder.
	X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
	X = conv_bn_relu(X_input, nbunits=768, kernel_size=3, kernel_regularizer=l2(l2_rate), padding='same')
	X = AveragePooling1D(pool_size=2, strides=2, padding='valid')(X)
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=1024, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	X = conv_bn_relu(X, nbunits=1024, kernel_size=1, kernel_regularizer=l2(l2_rate), padding='same')
	X = Dropout(dropout_rate)(X)
	
	#-- Flatten + 	1 FC layers
	X = GlobalAveragePooling1D()(X)
	for add in range(nb_fc):	
		X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
	out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.
	return Model(inputs = X_input, outputs = out, name='Archi_3CONV2AP_1FC256_GAP_f3_1_1fd')		
	
	

#-----------------------------------------------------------------------		
#-----------------------------------------------------------------------

#--------------------- Switcher for running the architectures
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	batch_size = 32
	
	switcher = {
		0: Archi_3CONV64C_1FC256_GAP_f3fd,
		1: Archi_3CONV64C_1FC256_GAP_f5fd,
		2: Archi_3CONV64C_1FC256_GAP_f9fd,
		3: Archi_3CONV64C_1FC256_GAP_f17fd,
		4: Archi_3CONV64C_1FC256_GAP_f33fd,
		
		10: Archi_3CONV2MP_1FC256_f33_17_9fd,
		11: Archi_3CONV2MP_1FC256_f17_9_5fd,
		12: Archi_3CONV2MP_1FC256_f9_5_3fd,
		13: Archi_3CONV2MP_1FC256_f5_3_1fd, 
		14: Archi_3CONV2MP_1FC256_f3_1_1fd, 
		
		15: Archi_3CONV2AP_1FC256_f33_17_9fd,
		16: Archi_3CONV2AP_1FC256_f17_9_5fd,
		17: Archi_3CONV2AP_1FC256_f9_5_3fd,
		18: Archi_3CONV2AP_1FC256_f5_3_1fd, 
		19: Archi_3CONV2AP_1FC256_f3_1_1fd, 
	
		20: Archi_3CONV2MP_1FC256_GAP_f33_17_9fd,
		21: Archi_3CONV2MP_1FC256_GAP_f17_9_5fd,
		22: Archi_3CONV2MP_1FC256_GAP_f9_5_3fd,
		23: Archi_3CONV2MP_1FC256_GAP_f5_3_1fd, 
		24: Archi_3CONV2MP_1FC256_GAP_f3_1_1fd, 
		
		25: Archi_3CONV2AP_1FC256_GAP_f33_17_9fd,
		26: Archi_3CONV2AP_1FC256_GAP_f17_9_5fd,
		27: Archi_3CONV2AP_1FC256_GAP_f9_5_3fd,
		28: Archi_3CONV2AP_1FC256_GAP_f5_3_1fd, 
		29: Archi_3CONV2AP_1FC256_GAP_f3_1_1fd					
	}
	func = switcher.get(noarchi, lambda: 0)
	model = func(args[0], args[1].shape[1])
	
	if len(args)==5:
		return trainTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	elif len(args)==7:
		return trainValTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)

#EOF
