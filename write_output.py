#!/usr/bin/python

import os, sys
import argparse
import random

import gdal, osr, ogr
from gdalconst import *
from PIL import Image, ImageOps

import keras
from keras.models import Model, load_model

from sits.readingsits import *
from outputfiles.save import *
from deeplearning.architecture_features import *

#-----------------------------------------------------------------------		
def main(model_path, test_file, result_file, proba, feature):
	
	#-- Checking the extension
	assert result_file.split('.')[-1]==test_file.split('.')[-1], "ERR: requires similar extension"
	file_type = result_file.split('.')[-1]
	
	#-- Parameters to set
	n_channels = 3 #-- NIR, R, G
	
	#-- Get the number of classes
	n_classes = getNoClasses(model_path)
	
	#-- Read min max values
	minMaxVal_file = '.'.join(model_path.split('.')[0:-1])
	minMaxVal_file = minMaxVal_file + '_minMax.txt'
	if os.path.exists(minMaxVal_file):
		min_per, max_per = read_minMaxVal(minMaxVal_file)
	else:
		assert False, "ERR: min-max values needs to be stored during training"
	
	#-- Downloading
	if file_type=="csv":
		X_test, polygon_ids_test, y_test = readSITSData(test_file)		
		X_test = addingfeat_reshape_data(X_test, feature, n_channels)		
		X_test =  normalizingData(X_test, min_per, max_per)
	elif file_type=="tif":
		#---- Get image info about gps coordinates for origin plus size pixels
		image = gdal.Open(test_file, gdal.GA_ReadOnly) #, NUM_THREADS=8
		geotransform = image.GetGeoTransform()
		originX = geotransform[0]
		originY = geotransform[3]
		spacingX = geotransform[1]
		spacingY = geotransform[5]
		r, c = image.RasterYSize, image.RasterXSize
		out_raster_SRS = osr.SpatialReference()
		out_raster_SRS.ImportFromWkt(image.GetProjectionRef())
		
		#-- Set up the characteristics of the output image
		driver = gdal.GetDriverByName('GTiff')
		out_map_raster = driver.Create(result_file, c, r, 1, gdal.GDT_Byte)
		out_map_raster.SetGeoTransform([originX, spacingX, 0, originY, 0, spacingY])
		out_map_raster.SetProjection(out_raster_SRS.ExportToWkt())
		out_map_band = out_map_raster.GetRasterBand(1)
		
		if proba==True:
			result_conf_file = '.'.join(result_file.split('.')[0:-1]) + 'conf_map.tif'
			out_confmap_raster = driver.Create(result_conf_file, c, r, n_classes, gdal.GDT_Float32)
			out_confmap_raster.SetGeoTransform([originX, spacingX, 0, originY, 0, spacingY])
			out_confmap_raster.SetProjection(out_raster_SRS.ExportToWkt())
			
		
	#---- Loading the model
	model = load_model(model_path)
	
	if file_type=="csv":
		p_test = model.predict(x=X_test)
		if not proba:
			p_test = p_test.argmax(axis=1)
		write_predictions_csv(result_file, p_test)
		
	elif file_type=="tif":	
		#convert gps corners into image (x,y)
		def gps_2_image_xy(x,y):
			return (x-originX)/spacingX,(y-originY)/spacingY
		def gps_2_image_p(point):
			return gps_2_image_xy(point[0],point[1])

		size_areaX = c # decrease the values if the tiff data cannot be in the memory, e.g. size_areaX = 10980, r =50 (get tiff BlockSize information for a nice setting)
		size_areaY = r
		x_vec = list(range(int(c/size_areaX)))
		x_vec = [x*size_areaX for x in x_vec]
		y_vec = list(range(int(r/size_areaY)))
		y_vec = [y*size_areaY for y in y_vec]
		x_vec.append(c)
		y_vec.append(r)

		for x in range(len(x_vec)-1):
			for y in range(len(y_vec)-1):	
				xy_top_left = (x_vec[x],y_vec[y])
				xy_bottom_right = (x_vec[x+1],y_vec[y+1])
				#---- now loading associated data
				xoff = xy_top_left[0]
				yoff = xy_top_left[1]
				xsize = xy_bottom_right[0]-xy_top_left[0]
				ysize = xy_bottom_right[1]-xy_top_left[1]
				X_test = image.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize) #, gdal.GDT_Float32
				#---- reshape the cube in a column vector
				X_test = X_test.transpose((1,2,0))
				sX = X_test.shape[0]
				sY = X_test.shape[1]
				X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1],X_test.shape[2])
				#---- pre-processing the data
				X_test = addingfeat_reshape_data(X_test, feature, n_channels)
				X_test = normalizingData(X_test, min_per, max_per)
				#---- saving the information
				p_img = model.predict(X_test)				
				y_test = p_img.argmax(axis=1)
				pred_array = y_test.reshape(sX,sY)
				out_map_band.WriteArray(pred_array, xoff=xoff, yoff=yoff)
				out_map_band.FlushCache()
				if proba==True:
					confpred_array = p_img.reshape(sX,sY,n_classes)
					for b in range(n_classes):
						out_confmap_band = out_confmap_raster.GetRasterBand(b+1)		
						out_confmap_band.WriteArray(confpred_array[:,:,b], xoff=xoff, yoff=yoff)
					out_confmap_band.FlushCache()

	
	

#-----------------------------------------------------------------------		
if __name__ == "__main__":
	try:
		if len(sys.argv) == 1:
			prog = os.path.basename(sys.argv[0])
			print('      '+sys.argv[0]+' [options]')
			print("     Help: ", prog, " --help")
			print("       or: ", prog, " -h")
			print("example 1: python %s --model_path path/to/model --test_file path/to/test.csv --result_file path/to/results/result.csv --proba" %sys.argv[0])
			sys.exit(-1)
		else:
			parser = argparse.ArgumentParser(description='Running deep learning architectures on SITS datasets')
			parser.add_argument('--model_path', dest='model_path',
								help='path to the trained model',
								default=None)
			parser.add_argument('--test_file', dest='test_file',
								help='file to classify (csv/tif)',
								default="csv")
			parser.add_argument('--result_file', dest='result_file',
								help='path where to store the output file (same extension than test_file)',
								default=None)
			parser.add_argument('--proba', dest='proba',
								help='if True probabilities, rather than class, are stored',
								default=False, action="store_true")
			parser.add_argument('--feat', dest='feature',
								help='used feature vector',
								default="SB")
			args = parser.parse_args()
			main(args.model_path, args.test_file, args.result_file, args.proba, args.feature)
			print("0")
	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)

#EOF
