from secrets import access_key, secret_access_key

import boto3
import os


client = boto3.client('s3', 
					  aws_access_key_id = access_key,
					  aws_secret_access_key = secret_access_key) 

for dirpath, dirnames, filenames in os.walk("data_sample"):
	for file in filenames:
		if '.jpg' in file:
			upload_file_bucket = 'p8-recognize-fruits-bucket'
			print(dirpath)
			upload_file_key = str(dirpath+'/'+file)
			print(upload_file_key)
			client.upload_file(dirpath+'/'+file, upload_file_bucket, upload_file_key)

