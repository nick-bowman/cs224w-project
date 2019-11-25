BUCKET_NAME="cs224w-project"

if [ $# -lt 2 ] ; then
    echo "Must provide a link to download and bucket folder to upload to!"
fi

download_link=$1
wget -q $download_link
filename=$(basename -- "$download_link")
if [ "$3" == "--gunzip" ] ; then
    echo "Decompressing $filename..."
    gunzip $filename
    filename="${filename%.*}"
fi

bucket_folder=$2
gcs_bucket_path="gs://$BUCKET_NAME/$bucket_folder"
gsutil -m mv "$filename" "$gcs_bucket_path"
