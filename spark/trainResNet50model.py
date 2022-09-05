import re
from typing import Iterator

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType,\
	IntegerType, FloatType, ArrayType, BinaryType
from pyspark.sql.functions import col, pandas_udf, PandasUDFType

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


def normalize_array(arr):
	return tf.keras.applications.resnet50.preprocess_input(
		arr.reshape([224, 224, 3]))


def create_model():
	# Charger ResNet50 pré-entraîné sur ImageNet et sans les couches fully-connected
	model = ResNet50(weights="imagenet",
					 include_top=False,
					 input_shape=(224, 224, 3))
	# Récupérer la sortie de ce réseau
	x = model.output

	# On entraîne seulement le nouveau classifieur et on ne ré-entraîne pas les autres couches :
	for layer in model.layers:
		layer.trainable = False

	# Ajouter la nouvelle couche fully-connected pour la classification à 2 classes
	predictions = Dense(2, activation='softmax')(x)

	# Définir le nouveau modèle
	new_model = Model(inputs=model.input, outputs=predictions)

	# Compiler le modèle
	new_model.compile(loss="categorical_crossentropy",
					  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
					  metrics=["accuracy"])

	#new_model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
	#          optimizer=tensorflow.keras.optimizers.Adam(),
	#          metrics=['accuracy'])

	return new_model


def train_model(spark, model, dataframe):

	print(type(dataframe))

	# Convert dataset to RDD
	rdd = to_simple_rdd(spark, X_train, y_train)

	# Train model
	spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
	spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)


@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
	model = ResNet50()
	for input_array in iterator:
		normalized_input = np.stack(input_array.map(normalize_array))
		preds = model.predict(normalized_input)
		yield pd.Series(list(preds))


def get_label(row, resize=True):
	return re.split('/', row.origin)[-2]

def set_label(dataframe_batch_iterator:
					 Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
	for dataframe_batch in dataframe_batch_iterator:
		dataframe_batch["label"] = dataframe_batch.apply(
			get_label, args=(True,), axis=1)
		yield dataframe_batch

def main():
	# Generating Spark Context
	#conf = SparkConf().setAppName('MachineCurve').setMaster('local[8]')
	#spark = SparkContext(conf=conf)

	# Load data in a spark dataframe
	spark = SparkSession.builder.getOrCreate()

	s3_url = "../data_sample/fruits-360_dataset/fruits-360/Training/"
	images_df = spark.read.format("image").option("recursiveFileLookup", "true").load(s3_url)

	print((images_df.count(), len(images_df.columns)))

	images_df.printSchema()

	# Add label in images_df
	schema = StructType(StructType(images_df.select("image.*").schema.fields + [
		StructField("label", StringType(), True)
	]))
	images_label_df = images_df.select("image.*").mapInPandas(set_label, schema)

	row = images_label_df.collect()[0]
	Image.frombytes(mode='RGB', data=bytes(row.data),
					size=[row.width, row.height]).show()

	# Resized images to be taken by ResNet50 model
	schema = StructType(images_label_df.select("*").schema.fields + [
		StructField("data_as_resized_array", ArrayType(IntegerType()), True),
		StructField("data_as_array", ArrayType(IntegerType()), True)
	])
	resized_df = images_label_df.select(
		"*").mapInPandas(resize_image_udf, schema)

	resized_df.printSchema()

	# Create ResNet model
	model = create_model()

	# Train model
	model_trained = train_model(spark, model, resized_df)

	print(type(model_trained))

	# Predict for an image
	predicted_df = resized_df.withColumn(
		"predictions", predict_batch_udf("data_as_resized_array"))

	prediction_row = predicted_df.collect()[image_row]

	print(tf.keras.applications.resnet50.decode_predictions(
		np.array(prediction_row.predictions).reshape(1, 1000), top=5
	))


if __name__ == "__main__":
	main()
