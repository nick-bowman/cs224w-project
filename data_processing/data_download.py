from bs4 import BeautifulSoup
import requests
import os
import subprocess

DATA_URL = "https://zenodo.org/record/2539424#.XbDPZOdKgWp"
DATA_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "WikiLinksGraph")

def download_data(url, save_folder_path):
    """
        Takes a url from which to download WikiLinksGraph data and saves it
        into the specified folder. The downloading is performed via scraping
        of the webpage for <link> tags specifying a .gz file to download. A
        subprocess is then spun up to run wget to do the actual download.
        
        Parameters
        ----------
        url : (str) The url for the webpage hosting the download links
        save_folder_path : Path to the folder to save the files
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

def main():
    download_data(DATA_URL, DATA_FOLDER_PATH)

if __name__ == "__main__":
    main()