import time
from contextlib import contextmanager

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterator

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType,\
	IntegerType, FloatType, ArrayType, BinaryType
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf
from pyspark.ml.linalg import Vectors, _convert_to_vector, VectorUDT
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler

import tensorflow.keras.applications.resnet50 as resnet
from keras.models import Model

from PIL import Image

RESNET_HEIGHT = 224
RESNET_WIDTH = 224


# Instantiate spark to be connected with aws
spark = SparkSession.builder.master("local[8]").getOrCreate()
spark._jsc.hadoopConfiguration().set('fs.s3a.access.key', '')
spark._jsc.hadoopConfiguration().set(
	'fs.s3a.secret.key', '')

# import model with weights
model = resnet.ResNet50(weights="imagenet",
						input_shape=(RESNET_WIDTH, RESNET_HEIGHT, 3))
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# broadcast weights of the initial model
weights_ = spark.sparkContext.broadcast(model.get_weights())


def model_():  # Create model and add boadcasting weights
	# Create model without weights
	model = resnet.ResNet50(weights=None)
	model = Model(inputs=model.inputs,
				  outputs=model.layers[-2].output)
	# Import weights from broadcasting weights
	model.set_weights(weights_.value)
	return model


@contextmanager
def timer(title):
	t0 = time.time()
	yield
	print("{} - done in {:.0f}s".format(title, time.time() - t0))


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
	model = model_()
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
		# Transform array in series
		series = pd.Series(list(features))
		yield series


def get_label(row):
	return re.split('/', row.origin)[-2]


def set_label(dataframe_batch_iterator:
			  Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["label"] = dataframe_batch.apply(
			get_label, axis=1)
		yield dataframe_batch


def array_to_string(my_list):
	return '[' + ','.join([str(elem) for elem in my_list]) + ']'


def main():
	with timer('main process'):
		# Load images with spark
		filePath = "s3a://p8-recognize-fruits-bucket/data_sample/fruits-360_dataset/fruits-360/Training/"
		images_sdf = spark.read.format("image").option(
			"recursiveFileLookup", "true").load(filePath)

		# Add label in images_sdf
		schema = StructType(images_sdf.select("image.*").schema.fields + [
			StructField("label", StringType(), True)
		])
		images_sdf = images_sdf.select(
			"image.*").mapInPandas(set_label, schema)

		# Preprocess images to be taken by ResNet50 model
		schema = StructType(images_sdf.schema.fields + [
			StructField("data_as_array", ArrayType(IntegerType()), True)
		])
		images_sdf = images_sdf.mapInPandas(prepro_image_udf, schema)

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


		sdf = images_sdf.select('origin', 'label', 'dense_features').join(
			pca_features.select('pca_features', 'dense_features'),
			on=["dense_features"]).drop("dense_features")

		array_to_string_udf = udf(array_to_string, StringType())
		sdf = sdf.withColumn(
			'pca_features', array_to_string_udf(sdf["pca_features"]))
		sdf.coalesce(1).write.format('csv')\
			.mode("overwrite").option('header', 'true')\
			.save("result")

	# VISUALIZATION
	with timer('Visu'):
		image_label = images_sdf.select("label").toPandas()
		label_map = dict(zip(set(image_label['label']),
							 range(len(image_label['label']))))
		image_label['int'] = image_label['label'].map(label_map)

		reduce_image_df = pd.DataFrame(
			pca_features.toPandas()["pca_features"].to_list(),
			columns=['feature1', 'feature2'])

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
		scatter = ax.scatter(reduce_image_df["feature1"],
							 reduce_image_df["feature2"],
							 c=image_label['int'], cmap='Set1')
		ax.set_title(
			'Image projection colored by categories', fontsize=20)
		plt.show()


if __name__ == "__main__":
	main()
