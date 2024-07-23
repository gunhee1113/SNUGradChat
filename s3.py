import boto3
import os

def presigned_url(bucket_name, key, expiration=3600):
    s3_client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    url = ""
    try:
        url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': key}, ExpiresIn=expiration)
    except Exception as e:
        pass
    return url

if __name__ == "__main__":
    bucket_name = os.getenv("S3_BUCKET_NAME")
    key = "docs/컴퓨터공학부/복수전공-졸업규정.pdf"
    url = presigned_url(bucket_name, key)
    print(url)

