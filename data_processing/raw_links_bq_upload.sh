BUCKET_NAME="cs224w-project"
DIRECTORY_NAME="RawWikiLinks"

PROJECT_ID="cs224w-final-project-256817"
DATASET="RawWikiLinks"
TABLE="Full"

python raw_links_bucket_upload.py

gcs_bucket_path="gs://$BUCKET_NAME/$DIRECTORY_NAME"
bq load --allow_jagged_rows --skip_leading_rows=1 --max_bad_records=5 --source_format=CSV "$DATASET.$TABLE" \
     "$gcs_bucket_path/*.csv"
