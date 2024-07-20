import boto3

def check_aws_credentials(profile_name='default'):
    try:
        session = boto3.Session(profile_name=profile_name)
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        print("AWS credentials are valid.")
        print("Account ID:", identity['Account'])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_aws_credentials()
