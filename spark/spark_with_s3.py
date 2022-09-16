
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[8]").getOrCreate()

spark._jsc.hadoopConfiguration().set('fs.s3a.access.key', '')
spark._jsc.hadoopConfiguration().set('fs.s3a.secret.key', '')

filePath = "s3a://p8-recognize-fruits-bucket/data_sample/fruits-360_dataset/fruits-360/Training/AppleBraeburn/0_100.jpg"

img_df = spark.read\
  .format("image")\
  .load(filePath)

img_df.printSchema()