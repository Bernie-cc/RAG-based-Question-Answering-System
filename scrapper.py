import datetime
import os
import re
import shutil
from typing import List, Set, Dict
from bs4 import BeautifulSoup, Tag
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSemanticPreservingSplitter
from fake_headers import Headers
from collections import deque
from urllib.parse import urljoin, urlparse

# ---------------------------
# Config
# ---------------------------
config = {
    'chunk_size'    :   500,
    'chunk_overlap' :   20,
    'splitter'      :   'RecursiveCharacterTextSplitter' # RecursiveCharacterTextSplitter, HTMLSemanticPreservingSplitter
}


# ---------------------------
# Header Generator
# ---------------------------
def gen_header():
    header = Headers(browser="chrome", os="win", headers=True)
    return header.generate()


# ---------------------------
# Validate text
# ---------------------------
def is_valid_text(text, threshold=0.9):
    """
    Checks if text is valid (not binary garbage) based on the ratio of alphanumeric characters.
    """
    if not text:
        return False
    
    if len(text) < 20:
        return False

    # Remove excessive special characters
    cleaned = re.sub(r'[^a-zA-Z0-9.,!?;:\s]', '', text)

    # Calculate ratio of valid characters
    valid_ratio = len(cleaned) / len(text)

    return valid_ratio > threshold


# ---------------------------
# Scrape HTML content (text and href links)
# ---------------------------
def load_and_split(base_url):

    # Load HTML content
    loader = WebBaseLoader(
        web_paths = [base_url],
            requests_kwargs = {"headers" : gen_header()},
        )

    # Format docs
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
        ("h5", "Header 5"),
        ("h6", "Header 6"),
        ("p", "paragraph"),
    ]

    # Choose splitter
    if config["splitter"] == 'HTMLSemanticPreservingSplitter':
        html_splitter = HTMLSemanticPreservingSplitter(headers_to_split_on=headers_to_split_on, max_chunk_size=config["chunk_size"])
        html_content = loader.load()
        html_content_str = html_content[0].page_content
        docs = html_splitter.split_text(html_content_str)
    elif config['splitter'] == 'RecursiveCharacterTextSplitter':
        splitter = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
        docs = loader.load_and_split(text_splitter=splitter)
        plain = loader.load()
        html = loader.scrape()

    texts=[]
    for d in docs:
        t = re.sub(r'\n+', ' ', d.page_content).strip()
        t = re.sub(r'\n+', '\n', d.page_content).strip()
        t = re.sub(r'\s+', ' ', t)
        if is_valid_text(t):
            texts.append(t)

    plain_texts=[]
    for p in plain:
        t = re.sub(r'\n+', ' ', d.page_content).strip()
        t = re.sub(r'\n+', '\n', d.page_content).strip()
        t = re.sub(r'\s+', ' ', t)
        if is_valid_text(t):
            plain_texts.append(t)

    # #self.visited_links.append(base_url)

    # # Extract links
    # urls = []
    # html = loader.scrape()
    # for a in html.find_all('a', href=True):
    #     href = a['href']
    #     if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
    #         full_url = urljoin(base_url, href)
    #         base_domain = urlparse(base_url).netloc
    #         link_domain = urlparse(full_url).netloc

    #         if base_domain == link_domain:
    #             urls.append(full_url)
    
    return texts, plain_texts

def ark_and_new(file_path):

    ark_folder = "_ark"
    os.makedirs(ark_folder, exist_ok=True)

    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")  # Generate timestamp
        new_filename = f"{timestamp}_{filename}"  # Append timestamp
        ark_path = os.path.join(ark_folder, new_filename)

        shutil.move(file_path, ark_path)
        print(f"Existing file moved to: {ark_path}")

    # Create a new empty file at the original location
    with open(file_path, "w", encoding="utf-8") as new_file:
        new_file.write("")  # Write an empty file or initialize content
    print(f"Empty file created: {file_path}")

# -------------
# Driver
# -------------
failed_links = []
file_path = "output.txt"
file_path_plain = "output_plain.txt"
ark_and_new(file_path)
ark_and_new(file_path_plain)

with open("failed_links.txt", "r", encoding="utf-8") as file:
    failed_links = [line.strip() for line in file]

for l in failed_links:
    texts, plain_texts = load_and_split(l)
    with open(file_path, "a", encoding="utf-8") as f:
        for chunks in texts:
            f.write(chunks + "\n\n")
    print(f"{l}: Saved {len(texts)} text chunks to file")

    with open(file_path_plain, "a", encoding="utf-8") as f:
        for chunks in plain_texts:
            f.write(chunks + "\n\n")
    print(f"{l}: Saved {len(plain_texts)} text blocks to file")