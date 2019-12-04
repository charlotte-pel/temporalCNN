#!/usr/bin/python

""" 
	Defining keras architecre, and training the models
"""

import sys, os
import numpy as np
import time

import keras
from keras import layers
from keras import optimizers
from keras.regularizers import l2
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Lambda, SpatialDropout1D, Concatenate
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import Callback, ModelCheckpoint, History, EarlyStopping
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras import backend as K



#-----------------------------------------------------------------------
#---------------------- Modules
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------		
def conv_bn(X, **conv_params):	
	nbunits = conv_params["nbunits"];
	kernel_size = conv_params["kernel_size"];

	strides = conv_params.setdefault("strides", 1)
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

	Z = Conv1D(nbunits, kernel_size=kernel_size, 
			strides = strides, padding=padding,
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer)(X)

	return BatchNormalization(axis=-1)(Z) #-- CHANNEL_AXIS (-1)

#-----------------------------------------------------------------------		
def conv_bn_relu(X, **conv_params):
	Znorm = conv_bn(X, **conv_params)
	return Activation('relu')(Znorm)
	
#-----------------------------------------------------------------------		
def conv_bn_relu_drop(X, **conv_params):	
	dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
	A = conv_bn_relu(X, **conv_params)
	return Dropout(dropout_rate)(A)

#-----------------------------------------------------------------------		
def conv_bn_relu_spadrop(X, **conv_params):	
	dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
	A = conv_bn_relu(X, **conv_params)
	return SpatialDropout1D(dropout_rate)(A)

#-----------------------------------------------------------------------		
def conv2d_bn(X, **conv_params):	
	nbunits = conv_params["nbunits"];
	kernel_size = conv_params["kernel_size"];

	strides = conv_params.setdefault("strides", 1)
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

	Z = Conv2D(nbunits, kernel_size=kernel_size, 
			strides = strides, padding=padding,
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer)(X)

	return BatchNormalization(axis=-1)(Z) #-- CHANNEL_AXIS (-1)

#-----------------------------------------------------------------------		
def conv2d_bn_relu(X, **conv_params):
	Znorm = conv2d_bn(X, **conv_params)
	return Activation('relu')(Znorm)
	
#-----------------------------------------------------------------------		
def conv2d_bn_relu_drop(X, **conv_params):	
	dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
	A = conv2d_bn_relu(X, **conv_params)
	return Dropout(dropout_rate)(A)

#-----------------------------------------------------------------------		
def conv2d_bn_relu_spadrop(X, **conv_params):	
	dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
	A = conv2d_bn_relu(X, **conv_params)
	return SpatialDropout1D(dropout_rate)(A)

	
#-----------------------------------------------------------------------		
def relu_drop(X, **conv_params):	
	dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
	A = Activation('relu')(X)
	return Dropout(dropout_rate)(A)

#-----------------------------------------------------------------------		
def fc_bn(X, **fc_params):
	nbunits = fc_params["nbunits"];
	
	kernel_regularizer = fc_params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = fc_params.setdefault("kernel_initializer", "he_normal")
		
	Z = Dense(nbunits, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(X)
	return BatchNormalization(axis=-1)(Z) #-- CHANNEL_AXIS (-1)
	
#-----------------------------------------------------------------------		
def fc_bn_relu(X, **fc_params):	
	Znorm = fc_bn(X, **fc_params)
	return Activation('relu')(Znorm)

#-----------------------------------------------------------------------		
def fc_bn_relu_drop(X, **fc_params):
	dropout_rate = fc_params.setdefault("dropout_rate", 0.5)
	A = fc_bn_relu(X, **fc_params)
	return Dropout(dropout_rate)(A)

#-----------------------------------------------------------------------		
def softmax(X, nbclasses, **params):
	kernel_regularizer = params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = params.setdefault("kernel_initializer", "glorot_uniform")
	return Dense(nbclasses, activation='softmax', 
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer)(X)

#-----------------------------------------------------------------------		
def getNoClasses(model_path):
	model = load_model(model_path)
	last_weight = model.get_weights()[-1]
	nclasses = last_weight.shape[0] #--- get size of the bias in the Softmax
	return nclasses

#-----------------------------------------------------------------------
#---------------------- Training models
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def trainTestModel(model, X_train, Y_train_onehot, X_test, Y_test_onehot, out_model_file, **train_params):
	#---- variables
	n_epochs = train_params.setdefault("n_epochs", 20)
	batch_size = train_params.setdefault("batch_size", 32)
	
	lr = train_params.setdefault("lr", 0.001)
	beta_1 = train_params.setdefault("beta_1", 0.9)
	beta_2 = train_params.setdefault("beta_2", 0.999)
	decay = train_params.setdefault("decay", 0.0)

	#---- optimizer
	opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
	model.compile(optimizer = opt, loss = "categorical_crossentropy",
			metrics = ["accuracy"])
	
	#---- monitoring the minimum loss
	checkpoint = ModelCheckpoint(out_model_file, monitor='loss',
			verbose=0, save_best_only=True, mode='min')
	callback_list = [checkpoint]
		
	start_train_time = time.time()
	hist = model.fit(x = X_train, y = Y_train_onehot, epochs = n_epochs, 
		batch_size = batch_size, shuffle=True,
		validation_data=(X_test, Y_test_onehot),
		verbose=1, callbacks=callback_list)
	train_time = round(time.time()-start_train_time, 2)
		
	#-- download the best model
	del model	
	model = load_model(out_model_file)
	start_test_time = time.time()
	test_loss, test_acc = model.evaluate(x=X_test, y=Y_test_onehot, 
		batch_size = 128, verbose=0)
	test_time = round(time.time()-start_test_time, 2)
	
	return test_acc, np.min(hist.history['loss']), model, hist.history, train_time, test_time

#-----------------------------------------------------------------------
def trainTestModel_EarlyAbandon(model, X_train, Y_train_onehot, X_test, Y_test_onehot, out_model_file, **train_params):
	#---- variables
	n_epochs = train_params.setdefault("n_epochs", 20)
	batch_size = train_params.setdefault("batch_size", 32)
	
	lr = train_params.setdefault("lr", 0.001)
	beta_1 = train_params.setdefault("beta_1", 0.9)
	beta_2 = train_params.setdefault("beta_2", 0.999)
	decay = train_params.setdefault("decay", 0.0)

	#---- optimizer
	opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
	model.compile(optimizer = opt, loss = "categorical_crossentropy",
			metrics = ["accuracy"])
	
	#---- monitoring the minimum loss
	checkpoint = ModelCheckpoint(out_model_file, monitor='loss',
			verbose=0, save_best_only=True, mode='min')
	early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto')
	callback_list = [checkpoint, early_stop]
			
	start_train_time = time.time()
	hist = model.fit(x = X_train, y = Y_train_onehot, epochs = n_epochs, 
		batch_size = batch_size, shuffle=True,
		validation_data=(X_test, Y_test_onehot),
		verbose=1, callbacks=callback_list)
	train_time = round(time.time()-start_train_time, 2)
		
	#-- download the best model
	del model	
	model = load_model(out_model_file)
	start_test_time = time.time()
	test_loss, test_acc = model.evaluate(x=X_test, y=Y_test_onehot, 
		batch_size = 128, verbose=0)
	test_time = round(time.time()-start_test_time, 2)
	
	return test_acc, np.min(hist.history['loss']), model, hist.history, train_time, test_time
		
#-----------------------------------------------------------------------
def trainValTestModel(model, X_train, Y_train_onehot, X_val, Y_val_onehot, X_test, Y_test_onehot, out_model_file, **train_params):
	#---- variables
	n_epochs = train_params.setdefault("n_epochs", 20)
	batch_size = train_params.setdefault("batch_size", 32)
	
	lr = train_params.setdefault("lr", 0.001)
	beta_1 = train_params.setdefault("beta_1", 0.9)
	beta_2 = train_params.setdefault("beta_2", 0.999)
	decay = train_params.setdefault("decay", 0.0)

	#---- optimizer
	opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
	model.compile(optimizer = opt, loss = "categorical_crossentropy",
			metrics = ["accuracy"])
	
	#---- monitoring the minimum validation loss
	checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
			verbose=0, save_best_only=True, mode='min')
	callback_list = [checkpoint]
		
	start_train_time = time.time()
	hist = model.fit(x = X_train, y = Y_train_onehot, epochs = n_epochs, 
		batch_size = batch_size, shuffle=True,
		validation_data=(X_val, Y_val_onehot),
		verbose=1, callbacks=callback_list)
	train_time = round(time.time()-start_train_time, 2)
		
	#-- download the best model
	del model	
	model = load_model(out_model_file)
	start_test_time = time.time()
	test_loss, test_acc = model.evaluate(x=X_test, y=Y_test_onehot, 
		batch_size = 128, verbose=0)
	test_time = round(time.time()-start_test_time, 2)
	
	return test_acc, np.min(hist.history['val_loss']), model, hist.history, train_time, test_time
	
#-----------------------------------------------------------------------
def trainValTestModel_EarlyAbandon(model, X_train, Y_train_onehot, X_val, Y_val_onehot, X_test, Y_test_onehot, out_model_file, **train_params):
	#---- variables
	n_epochs = train_params.setdefault("n_epochs", 20)
	batch_size = train_params.setdefault("batch_size", 32)
	
	lr = train_params.setdefault("lr", 0.001)
	beta_1 = train_params.setdefault("beta_1", 0.9)
	beta_2 = train_params.setdefault("beta_2", 0.999)
	decay = train_params.setdefault("decay", 0.0)

	#---- optimizer
	opt = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
	model.compile(optimizer = opt, loss = "categorical_crossentropy",
			metrics = ["accuracy"])
	
	#---- monitoring the minimum validation loss
	checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss',
			verbose=0, save_best_only=True, mode='min')
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
	callback_list = [checkpoint, early_stop]
		
	start_train_time = time.time()
	hist = model.fit(x = X_train, y = Y_train_onehot, epochs = n_epochs, 
		batch_size = batch_size, shuffle=True,
		validation_data=(X_val, Y_val_onehot),
		verbose=1, callbacks=callback_list)
	train_time = round(time.time()-start_train_time, 2)
		
	#-- download the best model
	del model	
	model = load_model(out_model_file)
	start_test_time = time.time()
	test_loss, test_acc = model.evaluate(x=X_test, y=Y_test_onehot, 
		batch_size = 128, verbose=0)
	test_time = round(time.time()-start_test_time, 2)
	
	return test_acc, np.min(hist.history['val_loss']), model, hist.history, train_time, test_time

#EOF
