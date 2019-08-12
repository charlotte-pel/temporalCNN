#!/usr/bin/python

""" 
	Some functions to read and compute spectral features on SITS
"""


import sys, os
import numpy as np
import pandas as pd
import math
import random
import itertools

#-----------------------------------------------------------------------
#---------------------- SATELLITE MODULE
#-----------------------------------------------------------------------
final_class_label = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12']


#-----------------------------------------------------------------------
def readSITSData(name_file):
	"""
		Read the data contained in name_file
		INPUT:
			- name_file: file where to read the data
		OUTPUT:
			- X: variable vectors for each example
			- polygon_ids: id polygon (use e.g. for validation set)
			- Y: label for each example
	"""
	
	data = pd.read_table(name_file, sep=',', header=None)
	
	y_data = data.iloc[:,0]
	y = np.asarray(y_data.values, dtype='uint8')
	
	polygonID_data = data.iloc[:,1]
	polygon_ids = polygonID_data.values
	polygon_ids = np.asarray(polygon_ids, dtype='uint16')
		
	X_data = data.iloc[:,2:]
	X = X_data.values
	X = np.asarray(X, dtype='float32')

	return  X, polygon_ids, y


#-----------------------------------------------------------------------
def addFeatures(X):
	"""
		Read the data contained in name_file
		INPUT:
			- X: orginal X features composed of threes bands (NIR-R-G)
				in the following order 
					[date1.NIR, date1.R, date1.G, ..., dateD.NIR, dateD.R, dateD.G]
		OUTPUT:
			- X_features: orginal_X with the addition of NDVI, NDWI and Brilliance
				in the following order	
					[X, date1.NDVI, ..., dateD.NDVI, date1.NDWI, ..., dateD.NDWI, date1.Brilliance, ..., dateD.Brilliance]
	"""
	n_channels = 3
	
	NIR = X[:,0::n_channels]
	NIR = np.array(NIR)
	NIR = NIR.astype(np.float)
	R = X[:,1::n_channels]
	R = np.array(R)
	R = R.astype(np.float)
	G = X[:,2::n_channels]
	G = np.array(G)
	G = G.astype(np.float)	
	
	NDVI = np.where(NIR+R!=0., (NIR-R)/(NIR+R), 0.)
	NDVI = NDVI.astype(float)
	
	
	NDWI = np.where(G+NIR!=0., (G-NIR)/(G+NIR), 0.)
	NDWI = NDWI.astype(float)
	
	Brilliance = np.sqrt((NIR*NIR + R*R + G*G)/3.0)
	Brilliance = Brilliance.astype(float)
	
	return NDVI, NDWI, Brilliance

#-----------------------------------------------------------------------
def computeNDVI(X, n_channels):
	"""
		Read the data contained in name_file
		INPUT:
			- X: orginal X features composed of threes bands (NIR-R-G)
				in the following order 
					[date1.NIR, date1.R, date1.G, ..., dateD.NIR, dateD.R, dateD.G]
		OUTPUT:
			- X_features: orginal_X with the addition of NDVI, NDWI and Brilliance
				in the following order	
					[X, date1.NDVI, ..., dateD.NDVI, date1.NDWI, ..., dateD.NDWI, date1.Brilliance, ..., dateD.Brilliance]
	"""
	
	NIR = X[:,0::n_channels]
	NIR = np.array(NIR)
	NIR = NIR.astype(np.float)
	R = X[:,1::n_channels]
	R = np.array(R)
	R = R.astype(np.float)
	
	NDVI = np.where(NIR+R!=0., (NIR-R)/(NIR+R), 0.)
	return NDVI

#-----------------------------------------------------------------------
def addingfeat_reshape_data(X, feature_strategy, nchannels):
	"""
		Reshaping (feature format (3 bands): d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...)
		INPUT:
			-X: original feature vector ()
			-feature_strategy: used features (options: SB, NDVI, SB3feat)
			-nchannels: number of channels
		OUTPUT:
			-new_X: data in the good format for Keras models
	"""
			
	if feature_strategy=='SB':
		print("SPECTRAL BANDS-----------------------------------------")
		return X.reshape(X.shape[0],int(X.shape[1]/nchannels),nchannels)
								
	elif feature_strategy=='NDVI':
		print("NDVI only----------------------------------------------")
		new_X = computeNDVI(X, nchannels)
		return np.expand_dims(new_X, axis=2)
							
	elif feature_strategy=='SB3feat':
		print("SB + NDVI + NDWI + IB----------------------------------")
		NDVI, NDWI, IB = addFeatures(X)		
		new_X = X.reshape(X.shape[0],int(X.shape[1]/nchannels),nchannels)		
		new_X = np.dstack((new_X, NDVI))
		new_X = np.dstack((new_X, NDWI))
		new_X = np.dstack((new_X, IB))
		return new_X
	else:
		print("Not referenced!!!-------------------------------------------")
		return -1

#-----------------------------------------------------------------------
def computingMinMax(X, per=2):
	min_per = np.percentile(X, per, axis=(0,1))
	max_per = np.percentile(X, 100-per, axis=(0,1))
	return min_per, max_per

#-----------------------------------------------------------------------
def normalizingData(X, min_per, max_per):
	return (X-min_per)/(max_per-min_per)

#-----------------------------------------------------------------------	
def extractValSet(X_train, polygon_ids_train, y_train, val_rate=0.1):
	unique_pol_ids_train, indices = np.unique(polygon_ids_train, return_inverse=True) #-- pold_ids_train = unique_pol_ids_train[indices]
	nb_pols = len(unique_pol_ids_train)
	
	ind_shuffle = list(range(nb_pols))
	random.shuffle(ind_shuffle)
	list_indices = [[] for i in range(nb_pols)]
	shuffle_indices = [[] for i in range(nb_pols)]
	[ list_indices[ind_shuffle[val]].append(idx) for idx, val in enumerate(indices)]					
		
	final_ind = list(itertools.chain.from_iterable(list_indices))
	m = len(final_ind)
	final_train = int(math.ceil(m*(1.0-val_rate)))
	
	shuffle_polygon_ids_train = polygon_ids_train[final_ind]
	id_final_train = shuffle_polygon_ids_train[final_train]
	
	while shuffle_polygon_ids_train[final_train-1]==id_final_train:
		final_train = final_train-1
	
	
	new_X_train = X_train[final_ind[:final_train],:,:]
	new_y_train = y_train[final_ind[:final_train]]
	new_X_val = X_train[final_ind[final_train:],:,:]
	new_y_val = y_train[final_ind[final_train:]]
	
	return new_X_train, new_y_train, new_X_val, new_y_val
	

#EOF
