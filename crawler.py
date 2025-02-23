import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def create_directories():
    """Create necessary data directories"""
    directories = ['local_data/raw_data', 'local_data/process_data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def download_html(url, raw_data_dir):
    """
    Download webpage HTML content and save locally
    
    Args:
        url: URL of the webpage to download
        raw_data_dir: Directory to save raw data
        
    Returns:
        tuple: (filename, HTML content)
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Generate filename from URL
        filename = urlparse(url).path.strip('/')
        filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        filename = f"{filename}.html"
        
        # Save HTML file
        filepath = os.path.join(raw_data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        return filename, response.text
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None, None

def extract_text_and_links(html_content, base_url):
    """
    Extract text and relevant links from HTML content
    
    Args:
        html_content: HTML content
        base_url: Base URL for building complete links
        
    Returns:
        tuple: (text content, list of relevant links)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Extract text
    text = soup.get_text(separator='\n', strip=True)
    
    # Extract relevant links
    links = []
    relevant_keywords = ['pittsburgh', 'carnegie', 'cmu', 'mellon']
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/wiki/') and ':' not in href:
            # Check if link text or href contains relevant keywords
            link_text = a.get_text().lower()
            href_lower = href.lower()
            
            is_relevant = any(keyword in link_text or keyword in href_lower 
                            for keyword in relevant_keywords)
            
            if is_relevant:
                full_url = urljoin(base_url, href)
                links.append(full_url)
    
    return text, links

def process_and_save_text(text, filename, process_data_dir):
    """
    Process and save extracted text
    
    Args:
        text: Text to process
        filename: Filename
        process_data_dir: Directory to save processed data
    """
    # Basic text cleaning
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
    
    # Save processed text
    processed_filename = filename.replace('.html', '.txt')
    filepath = os.path.join(process_data_dir, processed_filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def crawl_wikipedia(start_url, max_depth=1, max_subpages=30):
    """
    Main function to crawl Wikipedia pages
    
    Args:
        start_url: Starting URL for crawling
        max_depth: Maximum depth for crawling (default=1)
        max_subpages: Maximum number of subpages to crawl (default=30)
    """
    create_directories()
    visited_urls = set() 
    subpages_count = 0
    
    def crawl(url, depth=0):
        nonlocal subpages_count
        if depth > max_depth or url in visited_urls or subpages_count >= max_subpages:
            return
        
        visited_urls.add(url)
        print(f"Crawling: {url}")
        
        filename, html_content = download_html(url, 'local_data/raw_data')
        if html_content:
            text, links = extract_text_and_links(html_content, url)
            process_and_save_text(text, filename, 'local_data/process_data')
            
            # Count subpage only if it's not the start_url
            if url != start_url:
                subpages_count += 1
                print(f"Processed {subpages_count}/{max_subpages} subpages")
            
            # Recursively crawl linked pages
            if depth < max_depth:
                # Sort links to get most relevant ones first
                sorted_links = sorted(links, key=lambda x: sum(1 for keyword in ['pittsburgh', 'carnegie', 'cmu', 'mellon'] 
                                                             if keyword in x.lower()))
                sorted_links.reverse()  # Most relevant first
                
                for link in sorted_links:
                    if subpages_count < max_subpages:
                        crawl(link, depth + 1)
    
    crawl(start_url)

# Example usage
if __name__ == "__main__":
    wiki_url = "https://en.wikipedia.org/wiki/Pittsburgh"
    crawl_wikipedia(wiki_url, max_depth=1, max_subpages=50) 