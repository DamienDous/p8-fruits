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


def convert_bgr_array_to_rgb_array(img_array):
	B, G, R = img_array.T
	return np.array(((R+G+B)/3).T)

def resize_img(img_data):
	img = Image.frombytes(mode='RGB', data=img_data.data, size=[
		img_data.width, img_data.height])
	img = img.resize([224, 224], resample=Image.Resampling.BICUBIC) 
	arr = convert_bgr_array_to_rgb_array(np.asarray(img))
	arr = arr.reshape([224*224])
	return arr


def resize_image_udf(dataframe_batch_iterator:
					 Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["data_as_resized_array"] = dataframe_batch.apply(
			resize_img, axis=1)
		yield dataframe_batch


def normalize_array(arr):
	return tf.keras.applications.resnet50.preprocess_input(
		arr.reshape([224, 224]))


@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
	model = ResNet50()
	for input_array in iterator:
		normalized_input = np.stack(input_array.map(normalize_array))
		preds = model.predict(normalized_input)
		yield pd.Series(list(preds))


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

	row = image_df.select("image").collect()[0]
	Image.frombytes(mode='RGB', data=bytes(
		row.data), size=[row.width, row.height, 3]).show()

	# schema = StructType(image_df.select("image.*").schema.fields + [
	# 	StructField("data_as_resized_array", ArrayType(IntegerType()), True)
	# ])

	# resized_df = image_df.select(
	# 	"image.*").mapInPandas(resize_image_udf, schema)

	# row = resized_df.collect()[0]
	# Image.frombytes(mode='L', data=bytes(
	# 	row.data_as_resized_array), size=[224, 224]).show()

	# predicted_df = resized_df.withColumn(
	# 	"predictions", predict_batch_udf("data_as_resized_array"))

	# features_row = predicted_df.collect()[0]


	# print(features_row)

if __name__ == "__main__":
	main()
