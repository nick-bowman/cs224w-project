from bs4 import BeautifulSoup
import requests

def get_soup(link):
    """
    Takes a URL to a webpage and returns the BeautifulSoup object
    after parsing.

    Parameters
    ----------
    link : str
        Full URL to webpage.

    Returns
    -------
    soup : BeautifulSoup
        A BeautifulSoup object containing the parsed HTML.
    """
    response = requests.get(link)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    return soup
