"""Download RawWikiLinks dataset and immediately upload to a GCS bucket.

Usage:
  raw_links_bucket_upload.py [--language=<lang>] [--verbose]
  raw_links_bucket_upload.py (-h | --help)

Options:
  -h --help          Show this screen.
  --language=<lang>  Language to download. If not provided, all languages are downloaded.
  --verbose          Whether to include print statements.
"""

from docopt import docopt
import os
import subprocess
import utils
from concurrent.futures import ThreadPoolExecutor

HOME_FOLDER = os.path.expanduser("~")
WIKI_LINKS_RAW_URL = "http://cricca.disi.unitn.it/datasets/wikilinkgraphs-rawwikilinks/"
NUM_WORKERS = 8
BUCKET_FOLDER = "RawWikiLinks"

verbose = False # can be set to true via command line arguments (see main)

def save_and_upload_file(download_link):
    """
    Kicks off a shell script which downloads the RawWikiLinks file from the
    specified download link, decompresses the file, uploads the now-decompressed
    .csv file to the specified GCS bucket folder, and then removes the file from
    local machine.

    Parameters
    ----------
    download_link : str
        Link specifying the Raw WikiLinks file to download.
    """
    command = ["bash", "wget_upload_bucket.sh", download_link, BUCKET_FOLDER, "--gunzip"]
    if verbose:
        print(" ".join(command)) 
    subprocess.call(command)
    
def save_and_upload_language(language_link, num_workers = 1):
    """
    Iterates through the RawWikiLinks data for the given language and uploads
    each file to a GCS bucket.
    
    Parameters
    ----------
    language_link : str
        Link to the files storing the RawWikiLinks data for a single language.
    num_workers : int, optional
        Number of workers to use in the ThreadPool for parallelization, defaults
        to 1.
    """
    soup = utils.get_soup(language_link)
    file_names = soup.find_all("a", href = lambda tag : tag.endswith(".csv.gz"))
    if verbose:
        print("{} files to download...".format(len(file_names)))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_name in file_names:
            download_link = language_link + file_name.get("href")
            executor.submit(save_and_upload_file, (download_link))

def save_and_upload_all(download_url, num_workers = 1):
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
    for link in links:
        language_link = download_url + link.get("href") + "20180301/"
        if verbose:
            print("Downloading from {}...".format(language_link))
        save_and_upload_language(language_link, num_workers)
           

def main(args):
    global verbose
    verbose = bool(args["--verbose"])
    
    if verbose:
        print("Using {} threads for downloading.".format(NUM_WORKERS))
    
    if args["--language"]:
        lang = args["--language"]
        language_link = WIKI_LINKS_RAW_URL + "{}wiki/20180301/".format(lang)
        if verbose:
            print("Just downloading from {}...".format(language_link))
        save_and_upload_language(language_link, NUM_WORKERS)
    else:
        if verbose:
            print("Downloading all languages...")
        save_and_upload_all(WIKI_LINKS_RAW_URL, NUM_WORKERS)
    

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
