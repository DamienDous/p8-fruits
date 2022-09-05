import re
from typing import Iterator

from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, StringType,\
	IntegerType, FloatType, ArrayType, BinaryType
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.ml.feature import PCA

from PIL import Image
import pandas as pd
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import optimizers
import tensorflow as tf
from keras import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from pyspark import SparkContext, SparkConf
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from elephas.ml.adapter import to_data_frame

from pyspark.ml.linalg import Vectors


def get_label(row, resize=True):
	return re.split('/', row.origin)[-2]


def set_label(dataframe_batch_iterator:
			  Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["label"] = dataframe_batch.apply(
			get_label, args=(True,), axis=1)
		yield dataframe_batch


def create_model():
	# Charger ResNet50 pré-entraîné sur ImageNet et
	# sans les couches fully-connected
	model = ResNet50(weights="imagenet",
					 include_top=False,
					 input_shape=(224, 224, 3))
	# Récupérer la sortie de ce réseau
	x = model.output

	# On entraîne seulement le nouveau classifieur et
	# on ne ré-entraîne pas les autres couches :
	for layer in model.layers:
		layer.trainable = True

	# Ajouter la nouvelle couche fully-connected pour
	# la classification à 2 classes
	predictions = Dense(2, activation='softmax')(x)

	# Définir le nouveau modèle
	new_model = Model(inputs=model.input, outputs=predictions)

	# Compiler le modèle
	new_model.compile(loss="categorical_crossentropy",
					  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
					  metrics=["accuracy"])

	# new_model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
	#          optimizer=tensorflow.keras.optimizers.Adam(),
	#          metrics=['accuracy'])

	return new_model


def convert_bgr_array_to_rgb_array(img_array):
	B, G, R = img_array.T
	return np.array((R, G, B)).T


def resize_img(img_data, resize=True):
	mode = 'RGBA' if (img_data.nChannels == 4) else 'RGB'
	img = Image.frombytes(mode=mode, data=img_data.data, size=[
		img_data.width, img_data.height])
	img = img.convert('RGB') if (mode == 'RGBA') else img
	img = img.resize([224, 224], resample=Image.Resampling.BICUBIC) if (
		resize) else img
	arr = convert_bgr_array_to_rgb_array(np.asarray(img))
	arr = arr.reshape([224*224*3]) if (resize) else\
		arr.reshape([img_data.width*img_data.height*3])
	return arr


def resize_image_udf(dataframe_batch_iterator:
					 Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["data_as_resized_array"] = dataframe_batch.apply(
			resize_img, args=(True,), axis=1)
		dataframe_batch["data_as_array"] = dataframe_batch.apply(
			resize_img, args=(False,), axis=1)
		yield dataframe_batch


def preprocess_array(df):
	obj = tf.keras.applications.resnet50.preprocess_input(
		df.data_as_resized_array.reshape([224, 224, 3]))
	return obj.reshape([224*224*3])


def preproprocess_batch(dataframe_batch_iterator:
						Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["data_as_prepro_array"] = dataframe_batch.apply(
			preprocess_array, axis=1)
		yield dataframe_batch

def normalize_array(arr):
	return tf.keras.applications.resnet50.preprocess_input(
		arr.reshape([224, 224, 3]))

@pandas_udf(ArrayType(FloatType()))
def transform_batch_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
	# Create ResNet model
	model = ResNet50(weights="imagenet",
					 include_top=True,
					 input_shape=(224, 224, 3))
	print(model.summary())
	for input_array in iterator:
		print('input_array', type(input_array))
		print(input_array.shape)
		print(input_array[0].shape)
		print(input_array.map(normalize_array).shape)
		normalized_input = np.stack(input_array.map(normalize_array))
		print('normalized_input', type(normalized_input))
		print(normalized_input.shape)
		preds = model.predict(normalized_input)
		print('preds', type(preds))
		print(preds.shape)

		preds_output = preds.reshape(input_array.shape[0], 7*7*2048)

		print('preds_output', type(preds_output))
		print(preds_output.shape)
		liste = list(preds_output)
		series = pd.Series(liste)
		print(series.shape)
		yield series


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

	# Load data in a spark dataframe
	s3_url = "../data_sample/fruits-360_dataset/fruits-360/Training/"
	images_df = spark.read.format("image").option(
		"recursiveFileLookup", "true").load(s3_url)

	print((images_df.count(), len(images_df.columns)))

	# Add label in images_df
	schema = StructType(StructType(images_df.select("image.*").schema.fields + [
		StructField("label", StringType(), True)
	]))
	images_df = images_df.select("image.*").mapInPandas(set_label, schema)

	# Resized images to be taken by ResNet50 model
	schema = StructType(images_df.select("*").schema.fields + [
		StructField("data_as_resized_array", ArrayType(IntegerType()), True),
		StructField("data_as_array", ArrayType(IntegerType()), True)
	])
	images_df = images_df.mapInPandas(resize_image_udf, schema)

	# Preprocess images to be taken by ResNet50 model
	schema = StructType(images_df.select("*").schema.fields + [
		StructField("data_as_prepro_array", ArrayType(FloatType()), True)
	])
	images_df = images_df.select("*").mapInPandas(preproprocess_batch, schema)

	print(images_df.printSchema())

	# Predict for an image
	features_df = images_df.withColumn(
		"predictions", transform_batch_udf("data_as_resized_array"))

	print(features_df)

	pca = PCA(k=50, inputCol="features", outputCol="pcaFeatures")
	#pca_features_df = pca.fit(features_df['predictions'])

	# Mettre les predict df dans la base aws
	features_row = pca_features_df.collect()[0]

	#print(prediction_row.predictions)


		StructField("data_as_resized_array", ArrayType(IntegerType()), True),

if __name__ == "__main__":
	main()
