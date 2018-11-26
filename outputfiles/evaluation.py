#!/usr/bin/python

""" 
	Computing ML metrics for evaluating trained models
"""

import sys, os
import numpy as np
import math

#-----------------------------------------------------------------------		
def computingConfMatrix(referenced, p_test, n_classes):
	""" 
		Computing a n_classes by n_classes confusion matrix
		INPUT:
			- referenced: reference data labels
			- p_test: predicted 'probabilities' from the model for the test instances
			- n_classes: number of classes (numbered from 0 to 1)
		OUTPUT:
			- C: computed confusion matrix
	"""
	predicted = p_test.argmax(axis=1)
	C = np.zeros((n_classes, n_classes))
	for act, pred in zip(referenced, predicted):
		C[act][pred] += 1
	return C
	
#-----------------------------------------------------------------------		
def computingConfMatrixperPolygon(y_test, p_test, polygon_ids_test, n_classes):
	""" 
		Computing a n_classes by n_classes confusion matrix
		INPUT:
			- y_test_one_one: one hot encoding of the test labels
			- p_test: predicted 'probabilities' from the model for the test instances
			- n_classes: number of classes (numbered from 0 to 1)
		OUTPUT:
			- C_poly_perpoly: computed confusion matrix at polygon level with polygon count
			- C_poly_perpix: computed confusion matrix at polygon level with pixel count
			- OA_poly_poly: OA at polygon level with polygon count
			- OA_poly_pix: OA at polygon level with pixel count
	"""	
	nbTestInstances = y_test.shape[0]					
	unique_pol_test = np.unique(polygon_ids_test)
	n_polygons_test = len(unique_pol_test)
	C_poly_perpoly = np.zeros((n_classes, n_classes))
	C_poly_perpix = np.zeros((n_classes, n_classes))
	
	probas_per_polygon = {x:np.zeros(n_classes,dtype=float) for x in unique_pol_test}
	n_pixels_per_polygon = {x:0 for x in unique_pol_test}
	for i in range(nbTestInstances):
		poly = polygon_ids_test[i]
		pred = p_test[i]					
		probas_per_polygon[poly] = probas_per_polygon.get(poly) + pred
		n_pixels_per_polygon[poly] = n_pixels_per_polygon[poly] + 1

	for poly, probas in probas_per_polygon.items():
		probas_per_polygon[poly] = probas / n_pixels_per_polygon.get(poly)
		pred_class_id = np.argmax(probas_per_polygon[poly])
		id_line_with_right_poly = polygon_ids_test.tolist().index(poly)
		correct_class_index = y_test[id_line_with_right_poly]					
		C_poly_perpoly[correct_class_index,pred_class_id] = C_poly_perpoly[correct_class_index,pred_class_id] + 1
		C_poly_perpix[correct_class_index,pred_class_id] = C_poly_perpix[correct_class_index,pred_class_id] + n_pixels_per_polygon[poly]
	
	OA_poly_poly = round(float(np.trace(C_poly_perpoly))/n_polygons_test,4)
	OA_poly_pix = round(float(np.trace(C_poly_perpix))/nbTestInstances,4)
	return C_poly_perpoly, C_poly_perpix, OA_poly_poly, OA_poly_pix

	
#-----------------------------------------------------------------------		
def computingRMSE(y_test_one_hot,p_test):
	""""
		Computing RMSE from the prediction of the softmax layer
		INPUT:
			- y_test_one_one: one hot encoding of the test labels
			- p_test: predicted 'probabilities' from the model for the test instances
		OUTPUT:
			- rmse: Root Mean Square Error
	"""
	nbTestInstances = y_test_one_hot.shape[0]
	diff_proba = y_test_one_hot - p_test
	return math.sqrt(np.sum(diff_proba*diff_proba)/nbTestInstances)

#EOF
