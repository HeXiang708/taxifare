from google.cloud import storage

def download_blob(bucket_name: str,
                   source_blob_name: str,
                   destination_file_name: str
):
    """Downloads a blob from the bucket.

    Params:
    - bucket_name: str, name of the GCS bucket
    - source_blob_name: str, name of the blob in the bucket
    - destination_file_name: str, local path to save the downloaded file
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
