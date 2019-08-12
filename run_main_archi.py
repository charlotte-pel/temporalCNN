#!/usr/bin/python

import os, sys
import argparse
import random

from deeplearning.architecture_complexity import *
from outputfiles.save import *
from outputfiles.evaluation import *
from sits.readingsits import *


#-----------------------------------------------------------------------		
def main(sits_path, res_path, feature, noarchi, norun):
	
	#-- Creating output path if does not exist
	if not os.path.exists(res_path):
		os.makedirs(res_path)
	
	#---- Parameters to set
	n_channels = 3 #-- NIR, R, G
	val_rate = 0.05

	#---- Evaluated metrics
	eval_label = ['OA', 'train_loss', 'train_time', 'test_time']	
	
	#---- String variables
	train_str = 'train_dataset'
	test_str = 'test_dataset'					
	#---- Get filenames
	train_file = sits_path + '/' + train_str + '.csv'
	test_file = sits_path + '/' + test_str + '.csv'
	print("train_file: ", train_file)
	print("test_file: ", test_file)
	
	#---- output files			
	res_path = res_path + '/Archi' + str(noarchi) + '/'
	if not os.path.exists(res_path):
		os.makedirs(res_path)
	print("noarchi: ", noarchi)
	str_result = feature + '-' + train_str + '-noarchi' + str(noarchi) + '-norun' + str(norun) 
	res_file = res_path + '/resultOA-' + str_result + '.csv'
	res_mat = np.zeros((len(eval_label),1))
	traintest_loss_file = res_path + '/trainingHistory-' + str_result + '.csv'
	conf_file = res_path + '/confMatrix-' + str_result + '.csv'
	out_model_file = res_path + '/bestmodel-' + str_result + '.h5'

	#---- Downloading
	X_train, polygon_ids_train, y_train = readSITSData(train_file)
	X_test,  polygon_ids_test, y_test = readSITSData(test_file)
	
	n_classes_test = len(np.unique(y_test))
	n_classes_train = len(np.unique(y_train))
	if(n_classes_test != n_classes_train):
		print("WARNING: different number of classes in train and test")
	n_classes = max(n_classes_train, n_classes_test)
	y_train_one_hot = to_categorical(y_train, n_classes)
	y_test_one_hot = to_categorical(y_test, n_classes)			
	
	#---- Adding the features and reshaping the data if necessary
	X_train = addingfeat_reshape_data(X_train, feature, n_channels)
	X_test = addingfeat_reshape_data(X_test, feature, n_channels)		
		
	#---- Normalizing the data per band
	minMaxVal_file = '.'.join(out_model_file.split('.')[0:-1])
	minMaxVal_file = minMaxVal_file + '_minMax.txt'
	if not os.path.exists(minMaxVal_file): 
		min_per, max_per = computingMinMax(X_train)
		save_minMaxVal(minMaxVal_file, min_per, max_per)
	else:
		min_per, max_per = read_minMaxVal(minMaxVal_file)
	X_train =  normalizingData(X_train, min_per, max_per)
	X_test =  normalizingData(X_test, min_per, max_per)
	
	#---- Extracting a validation set (if necesary)
	if val_rate > 0:
		X_train, y_train, X_val, y_val = extractValSet(X_train, polygon_ids_train, y_train, val_rate)
		#--- Computing the one-hot encoding (recomputing it for train)
		y_train_one_hot = to_categorical(y_train, n_classes)
		y_val_one_hot = to_categorical(y_val, n_classes)

	if not os.path.isfile(res_file):
		if val_rate==0:
			res_mat[0,norun], res_mat[1,norun], model, model_hist, res_mat[2,norun], res_mat[3,norun] = \
				runArchi(noarchi, X_train, y_train_one_hot, X_test, y_test_one_hot, out_model_file)
		else:
			res_mat[0,norun], res_mat[1,norun], model, model_hist, res_mat[2,norun], res_mat[3,norun] = \
				runArchi(noarchi, X_train, y_train_one_hot, X_val, y_val_one_hot, X_test, y_test_one_hot, out_model_file)

		saveLossAcc(model_hist, traintest_loss_file)		
		p_test = model.predict(x=X_test)
		#---- computing confusion matrices
		C = computingConfMatrix(y_test, p_test,n_classes)
		#---- saving the confusion matrix
		save_confusion_matrix(C, final_class_label, conf_file)
				
		print('Overall accuracy (OA): ', res_mat[0,norun])
		print('Train loss: ', res_mat[1,norun])
		print('Training time (s): ', res_mat[2,norun])
		print('Test time (s): ', res_mat[3,norun])
		
		#---- saving res_file
		saveMatrix(np.transpose(res_mat), res_file, eval_label)

#-----------------------------------------------------------------------		
if __name__ == "__main__":
	try:
			main('./example', './example/res', 'SB', 2, 0)
			print("0")
	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)

#EOF
