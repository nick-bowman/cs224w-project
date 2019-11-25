"""Download WikiLinks data.

Usage:
  data_download.py

Options:
  -h --help     Show this screen.
"""

from docopt import docopt
from bs4 import BeautifulSoup
import requests
import os
import subprocess

HOME_FOLDER = os.path.expanduser("~")
WIKI_LINKS_GRAPH_URL = "https://zenodo.org/record/2539424#.XbDPZOdKgWp"

def download_wiki_links_graph(url, save_folder_path):
    """
        Takes a url from which to download WikiLinksGraph data and saves it
        into the specified folder. The downloading is performed via scraping
        of the webpage for <link> tags specifying a .gz file to download. A
        subprocess is then spun up to run wget to do the actual download.
        
        Parameters
        ----------
        url : str
            The url for the webpage hosting the download links
        save_folder_path : str
            Path to the folder to save the files
    """
    if not os.path.exists(save_folder_path):
        print("Creating folder to save data at {}".format(save_folder_path))
        os.makedirs(save_folder_path)

    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("link", type="application/gzip")
    command = ["wget", "-P", DATA_FOLDER_PATH, ""]
    for link in links:
        download_link = link.get("href")
        filename = download_link.split("/")[-1]
        file_path = os.path.join(DATA_FOLDER_PATH, filename)
        if os.path.exists(file_path):
            print("Skipping {}...".format(download_link))
            continue
        command[-1] = download_link
        print(" ".join(command)) 
        subprocess.call(command)

def main(args):
    save_folder_path = os.path.join(HOME_FOLDER, "WikiLinksGraph")
    download_wiki_links_graph(WIKI_LINKS_GRAPH_URL, save_folder_path)

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)