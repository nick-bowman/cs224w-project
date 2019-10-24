DIRECTORY_NAME="WikiLinksGraph"
BUCKET_NAME="cs224w-project"

directory_path="$HOME/$DIRECTORY_NAME"
if [ ! -d directory_path ]
then
    echo "Creating directory at $directory_path"
    mkdir "$directory_path"
fi
gcs_bucket_path="gs://$BUCKET_NAME/$DIRECTORY_NAME"
echo "About to download all files from $gcs_bucket_path to $directory_path..."
echo "Would recommend running this in a tmux session (takes ~45 minutes)."
echo "If you get credential/authentication errors, run \"gcloud auth login\" and try again."
gsutil -m cp -r "$gcs_bucket_path" "$directory_path"
