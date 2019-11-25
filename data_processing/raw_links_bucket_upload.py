import os
import subprocess
import utils
from concurrent.futures import ThreadPoolExecutor

HOME_FOLDER = os.path.expanduser("~")
WIKI_LINKS_RAW_URL = "http://cricca.disi.unitn.it/datasets/wikilinkgraphs-rawwikilinks/"
NUM_WORKERS = 4
BUCKET_FOLDER = "RawWikiLinks"

def save_and_upload_file(download_link):
    """
    Kicks off a shell script which downloads the RawWikiLinks file from the
    specified download link, decompresses the file, uploads the now-decompressed
    .csv file to the specified GCS bucket folder, and then removes the file from
    local machine.

    Parameters
    ----------
    download_link : str
        Link specifiying the Raw WikiLinks file to download.
    """
    command = ["bash", "wget_upload_bucket.sh", download_link, BUCKET_FOLDER, "--gunzip"]
    print(" ".join(command)) 
    subprocess.call(command) 

def save_and_upload(download_url, num_workers = 1):
    """
    Iterates through all of the RawWikiLinks data for each language and uploads
    each file to a GCS bucket.

    Parameters
    ----------
    download_url : str
        Link to the folders storing each of the RawWikiLinks data for each
        language.
    num_workers : int, optional
        Number of workers to use in the ThreadPool for parallelization, defaults
        to 1.
    """
    soup = utils.get_soup(download_url)
    links = soup.find_all("a", href = lambda tag : tag.endswith("wiki/"))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for link in links:
            folder_link = WIKI_LINKS_RAW_URL + link.get("href") + "20180301/"
            soup = utils.get_soup(folder_link)
            file_names = soup.find_all("a", href = lambda tag : tag.endswith(".csv.gz"))
            for file_name in file_names:
                download_link = folder_link + file_name.get("href")
                executor.submit(save_and_upload_file, (download_link))

def main():
    save_and_upload(WIKI_LINKS_RAW_URL, NUM_WORKERS)
    

if __name__ == "__main__":
    main()
