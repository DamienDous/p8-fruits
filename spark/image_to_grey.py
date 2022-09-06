from typing import Iterator

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType,\
	IntegerType, FloatType, ArrayType, BinaryType
from pyspark.sql.functions import col, pandas_udf, PandasUDFType

from PIL import Image
import pandas as pd
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf

RESNET50_IMG_SIZE = 224


def convert_bgr_array_to_rgb_array(img_array):
	print(img_array.shape)
	B, G, R = img_array.T
	res = ((R.astype(np.float32) +
			G.astype(np.float32) +
			B.astype(np.float32))/3).astype(int)
	print(res.T.shape)
	return res.T


def to_grey_img(img_data):
	img = Image.frombytes(mode='RGB', data=bytes(
		img_data.data), size=[img_data.width, img_data.height])
	grey_arr = convert_bgr_array_to_grey_array(np.asarray(img))
	grey_arr = grey_arr.reshape([img_data.width*img_data.height])
	return grey_arr


def to_grey_image_udf(dataframe_batch_iterator:
					  Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["data_as_grey_array"] = dataframe_batch.apply(
			to_grey_img, axis=1)
		yield dataframe_batch


def resize_img(img_data):
	#image = Image.frombytes(mode='L',
	#						data=bytes(img_data.data_as_grey_array),
	#						size=[img_data.width, img_data.height])
	#print(image.size)
	#image.show()
	reshape_array = img_data.data_as_grey_array.reshape((img_data.width, img_data.height), order='F')
	img = Image.fromarray(reshape_array, mode='L')
	img.show()

	img = img.resize((RESNET50_IMG_SIZE, RESNET50_IMG_SIZE),
					 resample=Image.Resampling.BICUBIC)
	return np.asarray(img).reshape((RESNET50_IMG_SIZE*RESNET50_IMG_SIZE))


def resize_image_udf(dataframe_batch_iterator:
					 Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["data_as_resized_array"] = dataframe_batch.apply(
			resize_img, axis=1)
		dataframe_batch["width"] = RESNET50_IMG_SIZE
		dataframe_batch["height"] = RESNET50_IMG_SIZE
		yield dataframe_batch


def main():
	# Generating Spark Context
	spark = (SparkSession.builder.master("local[1]")
			 .config("spark.driver.memory", "20g")
			 .config("spark.executor.memory", "10g")
			 .config("spark.driver.cores", "30")
			 .config("spark.num.executors", "8")
			 .config("spark.executor.cores", "4")
			 .getOrCreate())
	sc = spark.sparkContext

	s3_url = "../data_sample/fruits-360_dataset/fruits-360/Training/AppleBraeburn/"
	image_df = spark.read.format("image").load(s3_url)

	schema = StructType(image_df.select("image.*").schema.fields + [
		StructField("data_as_grey_array", ArrayType(IntegerType()), True)
	])
	grey_df = image_df.select(
		"image.*").mapInPandas(to_grey_image_udf, schema)
	row = grey_df.collect()[0]
	image = Image.frombytes(mode='L',
							data=bytes(row.data_as_grey_array),
							size=[row.width, row.height])
	print(image.size)
	image.show()

	schema = StructType(grey_df.schema.fields + [
		StructField("data_as_resized_array", ArrayType(IntegerType()), True)
	])
	resized_df = grey_df.mapInPandas(resize_image_udf, schema)

	row = resized_df.collect()[0]
	image = Image.frombytes(mode='L',
							data=bytes(row.data_as_resized_array),
							size=[row.width, row.height])
	print(image.size)
	#image.show()


if __name__ == "__main__":
	main()
