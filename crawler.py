import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import List, Set, Dict
from tqdm import tqdm
import PyPDF2
import io
import json
from pathlib import Path
from collections import deque

def create_directories():
    """Create necessary data directories"""
    directories = ['local_data/raw_data', 'local_data/process_data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

class WikiCrawler:
    def __init__(self, 
                 initial_urls: List[str],
                 max_depth: int = 1,
                 max_pages_per_url: int = 10,
                 max_total_pages: int = 50):
        """
        Initialize crawler with constraints and configurations
        
        Args:
            initial_urls: List of starting URLs to crawl
            max_depth: Maximum crawling depth from each initial URL
            max_pages_per_url: Maximum pages to crawl from each initial URL
            max_total_pages: Maximum total pages to crawl across all initial URLs
            
        The crawler is configured with:
        - Allowed domains for crawling
        - Content quality thresholds
        - URL and content validation rules
        - Metadata tracking
        """
        self.initial_urls = initial_urls
        self.max_depth = max_depth
        self.max_pages_per_url = max_pages_per_url
        self.max_total_pages = max_total_pages
        
        self.visited_urls = set()
        self.total_pages_crawled = 0
        self.pages_per_initial_url = {url: 0 for url in initial_urls}
        
        self.metadata = {
            'initial_urls': {},
            'total_size_mb': 0,
            'total_pages': 0
        }
        
        # List of allowed domains for crawling
        self.allowed_domains = [
            'wikipedia.org', 
            'cmu.edu', 
            'pittsburghpa.gov',
            'britannica.com',
            'visitpittsburgh.com'
        ]
        
        # Content quality thresholds
        self.min_word_count = 100  # Minimum words required in content
        self.max_newline_ratio = 0.5  # Maximum ratio of newlines to content length
        
    def download_html(self, url: str, raw_data_dir: str) -> tuple:
        """Download webpage HTML content and save locally"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filename = urlparse(url).path.strip('/')
            filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            filename = f"{filename}.html"
            
            filepath = os.path.join(raw_data_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            return filename, response.text
        
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None, None

    def download_and_process_pdf(self, url: str, raw_data_dir: str, process_data_dir: str) -> tuple:
        """
        Download and process PDF content
        
        Args:
            url: URL of the PDF file
            raw_data_dir: Directory to save raw PDF
            process_data_dir: Directory to save processed text
            
        Returns:
            tuple: (filename, text content)
        """
        try:
            # Download PDF
            response = requests.get(url)
            response.raise_for_status()
            
            # Generate filenames
            filename = urlparse(url).path.split('/')[-1]
            raw_filename = filename
            processed_filename = filename.replace('.pdf', '.txt')
            
            # Save raw PDF
            raw_filepath = os.path.join(raw_data_dir, raw_filename)
            with open(raw_filepath, 'wb') as f:
                f.write(response.content)
            
            # Extract text from PDF
            pdf_text = ""
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() + "\n"
            
            # Save processed text
            processed_filepath = os.path.join(process_data_dir, processed_filename)
            with open(processed_filepath, 'w', encoding='utf-8') as f:
                f.write(pdf_text)
            
            return raw_filename, pdf_text
            
        except Exception as e:
            print(f"Error processing PDF {url}: {e}")
            return None, None

    def extract_text_and_links(self, html_content: str, base_url: str) -> tuple:
        """
        Extract text and relevant links from HTML content
        
        Args:
            html_content: Raw HTML content to process
            base_url: Base URL for resolving relative links
            
        Returns:
            tuple: (extracted text, list of relevant links)
            
        Features:
        - Removes unwanted HTML elements
        - Extracts clean text content
        - Filters and validates links based on domain rules
        - Smart relevancy checking based on URL and content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        links = []
        relevant_keywords = ['pittsburgh', 'carnegie mellon university', 'cmu']
        
        def is_relevant_by_url(url: str) -> bool:
            """Check if URL itself contains relevant keywords"""
            url_lower = url.lower()
            return any(keyword in url_lower for keyword in relevant_keywords)
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Handle different URL patterns based on the base URL
            if 'wikipedia.org' in base_url:
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin("https://en.wikipedia.org", href)
                    # If URL contains keywords, add directly
                    if is_relevant_by_url(full_url):
                        links.append(full_url)
                    else:
                        # Otherwise check link text
                        link_text = a.get_text().lower()
                        if any(keyword in link_text for keyword in relevant_keywords):
                            links.append(full_url)
                        
            elif 'cmu.edu' in base_url:
                if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    full_url = urljoin(base_url, href)
                    if 'cmu.edu' in full_url:  # Only include CMU domain links
                        # CMU domain is already relevant, no need for keyword check
                        links.append(full_url)
                        
            elif 'pittsburghpa.gov' in base_url:
                if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    full_url = urljoin(base_url, href)
                    if 'pittsburghpa.gov' in full_url:  # Pittsburgh gov domain is relevant
                        links.append(full_url)
                        
            else:
                if not href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    full_url = urljoin(base_url, href)
                    base_domain = urlparse(base_url).netloc
                    link_domain = urlparse(full_url).netloc
                    
                    if base_domain == link_domain:
                        # If URL contains keywords, add directly
                        if is_relevant_by_url(full_url):
                            links.append(full_url)
                        else:
                            # Otherwise check link text
                            link_text = a.get_text().lower()
                            if any(keyword in link_text for keyword in relevant_keywords):
                                links.append(full_url)
        
        return text, links

    def process_and_save_text(self, text: str, filename: str, process_data_dir: str):
        """Process and save extracted text"""
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        processed_filename = filename.replace('.html', '.txt')
        filepath = os.path.join(process_data_dir, processed_filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

    def is_valid_url(self, url: str) -> bool:
        """
        Validate if a URL should be crawled based on multiple criteria
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL passes all validation checks, False otherwise
            
        Validation checks:
        1. File extension not in blocked list (images, scripts, etc.)
        2. Domain is in allowed list
        3. URL path not in excluded patterns (login, search, etc.)
        """
        parsed = urlparse(url)
        
        # Check file extensions
        invalid_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.css', '.js',
            '.xml', '.rss', '.zip', '.tar', '.gz'
        }
        if any(parsed.path.lower().endswith(ext) for ext in invalid_extensions):
            return False
            
        # Validate domain against allowlist
        if not any(domain in parsed.netloc for domain in self.allowed_domains):
            return False
            
        # Check for invalid URL patterns
        invalid_patterns = [
            '/action/', '/special:', '/category:', 
            '/file:', '/template:', '/help:', 
            'login', 'logout', 'register', 'search'
        ]
        if any(pattern in parsed.path.lower() for pattern in invalid_patterns):
            return False
            
        return True
        
    def validate_content(self, text: str, html_content: str = None) -> bool:
        """
        Validate content quality using multiple metrics
        
        Args:
            text: Extracted text content to validate
            html_content: Original HTML content (optional, no longer used)
            
        Returns:
            bool: True if content meets quality standards, False otherwise
            
        Quality checks:
        1. Minimum word count
        2. Newline ratio within limits
        3. Paragraph structure
        4. Complete sentences count
        """
        # Check text length
        words = text.split()
        if len(words) < self.min_word_count:
            print(f"Content too short: {len(words)} words")
            return False
            
        # Check newline ratio
        newline_ratio = text.count('\n') / len(text) if text else 1
        if newline_ratio > self.max_newline_ratio:
            print(f"Too many newlines: ratio {newline_ratio:.2f}")
            return False
            
        # Validate content structure
        # Check paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            print("No valid paragraphs found")
            return False
            
        # Verify complete sentences
        sentences = sum(1 for p in paragraphs for sent in p.split('.') if len(sent.strip()) > 20)
        if sentences < 3:
            print(f"Too few complete sentences: {sentences}")
            return False
            
        return True

    def crawl_url_bfs(self, start_url: str):
        """
        Perform breadth-first crawling starting from a given URL
        
        Args:
            start_url: Initial URL to start crawling from
            
        Process:
        1. Initialize queue with start URL
        2. Process URLs level by level
        3. Validate URLs and content
        4. Extract and process content
        5. Add new valid URLs to queue
        
        Features:
        - URL validation before processing
        - Content quality validation
        - PDF handling
        - Progress tracking
        - Depth control
        """
        queue = deque([(start_url, 0)])
        
        while queue and self.total_pages_crawled < self.max_total_pages:
            url, depth = queue.popleft()
            
            # 添加URL验证
            if not self.is_valid_url(url):
                continue
                
            if (depth > self.max_depth or 
                url in self.visited_urls or 
                self.pages_per_initial_url[start_url] >= self.max_pages_per_url):
                continue
            
            self.visited_urls.add(url)
            print(f"Crawling: {url} (Depth: {depth})")
            
            if url.lower().endswith('.pdf'):
                filename, content = self.download_and_process_pdf(
                    url, 'local_data/raw_data', 'local_data/process_data'
                )
                if content and self.validate_content(content):
                    self.total_pages_crawled += 1
                    self.pages_per_initial_url[start_url] += 1
                continue
            
            filename, html_content = self.download_html(url, 'local_data/raw_data')
            if html_content:
                text, links = self.extract_text_and_links(html_content, url)
                
                # 添加内容验证
                if self.validate_content(text, html_content):
                    self.process_and_save_text(text, filename, 'local_data/process_data')
                    self.total_pages_crawled += 1
                    self.pages_per_initial_url[start_url] += 1
                    
                    if depth < self.max_depth:
                        for link in links:
                            if link not in self.visited_urls:
                                queue.append((link, depth + 1))

    def calculate_file_sizes(self) -> Dict:
        """
        Calculate sizes of downloaded files for each initial URL
        
        Returns:
            Dict containing file size information
        """
        raw_data_path = Path('local_data/raw_data')
        processed_data_path = Path('local_data/process_data')
        
        # Initialize size tracking for each initial URL
        url_files = {url: {
            'raw_files': [],
            'processed_files': [],
            'raw_size_mb': 0,
            'processed_size_mb': 0,
            'subpages': self.pages_per_initial_url[url]
        } for url in self.initial_urls}
        
        # Calculate sizes for raw files
        for file_path in raw_data_path.glob('*'):
            size_mb = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
            
            # Find which initial URL this file belongs to
            for url in self.initial_urls:
                if url in str(file_path):
                    url_files[url]['raw_files'].append(str(file_path.name))
                    url_files[url]['raw_size_mb'] += size_mb
                    break
        
        # Calculate sizes for processed files
        for file_path in processed_data_path.glob('*'):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            for url in self.initial_urls:
                if url in str(file_path):
                    url_files[url]['processed_files'].append(str(file_path.name))
                    url_files[url]['processed_size_mb'] += size_mb
                    break
        
        return url_files

    def save_metadata(self):
        """Save metadata about the crawling process"""
        url_files = self.calculate_file_sizes()
        
        metadata = {
            'crawl_statistics': {
                'total_pages_crawled': self.total_pages_crawled,
                'max_depth_allowed': self.max_depth,
                'max_pages_per_url_allowed': self.max_pages_per_url,
                'max_total_pages_allowed': self.max_total_pages
            },
            'initial_urls': {}
        }
        
        total_raw_size = 0
        total_processed_size = 0
        
        # Compile metadata for each initial URL
        for url, stats in url_files.items():
            metadata['initial_urls'][url] = {
                'subpages_crawled': stats['subpages'],
                'raw_data': {
                    'size_mb': round(stats['raw_size_mb'], 2),
                    'files': stats['raw_files']
                },
                'processed_data': {
                    'size_mb': round(stats['processed_size_mb'], 2),
                    'files': stats['processed_files']
                },
                'total_size_mb': round(stats['raw_size_mb'] + stats['processed_size_mb'], 2)
            }
            total_raw_size += stats['raw_size_mb']
            total_processed_size += stats['processed_size_mb']
        
        # Add total sizes
        metadata['total_statistics'] = {
            'total_raw_size_mb': round(total_raw_size, 2),
            'total_processed_size_mb': round(total_processed_size, 2),
            'total_size_mb': round(total_raw_size + total_processed_size, 2)
        }
        
        # Save metadata to file
        with open('local_data/crawl_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nMetadata saved to local_data/crawl_metadata.json")

    def crawl_all(self):
        """Crawl all initial URLs with progress tracking using BFS"""
        create_directories()
        
        for url in tqdm(self.initial_urls, desc="Processing initial URLs"):
            print(f"\nStarting crawl from: {url}")
            self.crawl_url_bfs(url)
            
        print(f"\nCrawling completed:")
        print(f"Total pages crawled: {self.total_pages_crawled}")
        for url, count in self.pages_per_initial_url.items():
            print(f"Pages from {url}: {count}")
        
        # Save metadata after crawling
        self.save_metadata()

# Example usage
if __name__ == "__main__":
    initial_urls = [
        "https://en.wikipedia.org/wiki/Pittsburgh",
        "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
        "https://www.pittsburghpa.gov/Home",
        "https://www.britannica.com/place/Pittsburgh", 
        "https://www.visitpittsburgh.com/",
        "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms", 
        "https://www.cmu.edu/about/",
        "https://apps.pittsburghpa.gov/redtail/images/23255_2024_Operating_Budget.pdf"
         # Add more URLs here
    ]
    
    crawler = WikiCrawler(
        initial_urls=initial_urls,
        max_depth=4,
        max_pages_per_url=1000,
        max_total_pages=10000
    )
    
    crawler.crawl_all() 