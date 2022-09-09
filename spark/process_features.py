
import pandas as pd
import numpy as np

from typing import Iterator

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType,\
	IntegerType, FloatType, ArrayType, BinaryType
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf
from pyspark.ml.linalg import Vectors, _convert_to_vector, VectorUDT
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler

import scipy.sparse

from sklearn.decomposition import IncrementalPCA
import tensorflow.keras.applications.resnet50 as resnet
import tensorflow as tf

from PIL import Image

RESNET_HEIGHT = 224
RESNET_WIDTH = 224


def convert_bgr_to_rgb(img_array):
	B, G, R = img_array.T
	return np.array((R, G, B)).T


def image_preprocess(img_data):
	# Convert 1-dimensional array to 2 dimensional array
	img = Image.frombytes(mode='RGB', data=bytes(
		img_data.data), size=[img_data.width, img_data.height])
	# Resize image to be taken by resnet50
	resized_img = img.resize((RESNET_WIDTH, RESNET_HEIGHT),
							 resample=Image.Resampling.BICUBIC)
	# Transform BGR image to RGB array
	rgb_arr = convert_bgr_to_rgb(np.asarray(resized_img))
	# Apply Resnet50 preprocess on array
	#prepro_arr = resnet.preprocess_input(rgb_arr)
	# Reshape image array as 1-dimensional array
	arr_output = rgb_arr.reshape([RESNET_WIDTH*RESNET_HEIGHT*3])
	return arr_output


def prepro_image_udf(dataframe_batch_iterator:
					 Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	# Apply image preprocess on each batch
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["data_as_array"] = dataframe_batch.apply(
			image_preprocess, axis=1)
		yield dataframe_batch


@pandas_udf(ArrayType(FloatType()))
def process_features_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
	# Create ResNet model
	model = resnet.ResNet50(weights="imagenet",
							include_top=False,
							input_shape=(RESNET_WIDTH, RESNET_HEIGHT, 3))
	# Apply ResNet50 features process on each batch
	for series_batch in iterator:
		# Reshape batch array to be process by ResNet50 model
		array_batch = series_batch.map(
			lambda arr: arr.reshape([RESNET_WIDTH, RESNET_HEIGHT, 3]))
		# Apply Resnet50 preprocess on array batch
		res_prepro_batch = array_batch.map(
			lambda arr: resnet.preprocess_input(arr))
		# Transform serie to array
		tensor_batch = np.stack(res_prepro_batch)
		# Predict features with ResNet50
		features = model.predict(tensor_batch)
		# Reshape features as series_length*1-dimensional array
		features_output = features.reshape(features.shape[0],
										   features.shape[1] *
										   features.shape[2] *
										   features.shape[3])
		# Transform array in series
		series = pd.Series(list(features_output))
		yield series




def main():
	# Generating Spark Context
	spark = (SparkSession.builder.master("local[1]").getOrCreate())

	# Load images with spark
	s3_url = "../data_sample/fruits-360_dataset/fruits-360/Training/"
	images_df = spark.read.format("image").option(
		"recursiveFileLookup", "true").load(s3_url)

	# Preprocess images to be taken by ResNet50 model
	schema = StructType(images_df.select("image.*").schema.fields + [
		StructField("data_as_array", ArrayType(IntegerType()), True)
	])
	image_arrays_df = images_df.select(
		"image.*").mapInPandas(prepro_image_udf, schema)

	# Get ResNet50 features
	images_sdf = images_sdf.withColumn(
		"features", process_features_udf("data_as_array"))

	to_dense = udf(lambda vs: Vectors.dense(vs), VectorUDT())
	images_sdf = images_sdf.withColumn(
		"dense_features", to_dense(col("features")))

	# Apply PCA to dense_features column
	pca = PCA(k=2, inputCol="dense_features")
	pca.setOutputCol("pca_features")
	model = pca.fit(images_sdf.select("dense_features"))
	pca_features = model.transform(images_sdf.select("dense_features"))

	# # Print columns type
	# for col in features_df.dtypes:
	# 	print(col[0]+" , "+col[1])

	# Process and print features


if __name__ == "__main__":
	main()
