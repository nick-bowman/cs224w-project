PROJECT_ID="cs224w-final-project-256817"
DATASET="RawWikiLinks"
TABLE="Full"

if [ "$1" == "" ]; then
    echo "You must provide a path to a file to upload to BigQuery!"
    exit 1
fi

filename=$1
if [[ $filename == *.gz ]]; then
    echo "Unzipping $filename, may take a while..."
    gunzip $filename
    filename="${filename%.*}"
fi


if [[ $filename != *.csv ]]; then
    echo "File must be .csv! Exiting..."
    exit 1
fi

bq load \
--source_format=CSV --skip_leading_rows=1 \
$PROJECT_ID:$DATASET.$TABLE \
   $filename
