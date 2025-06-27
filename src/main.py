import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
from collections import deque

class WebCrawler:
    def __init__(self, base_url, max_pages=100, delay=1):
        """
        Initialize the web crawler
        
        Args:
            base_url: Starting URL to crawl
            max_pages: Maximum number of pages to crawl
            delay: Delay between requests (seconds) to be respectful
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls = set()
        self.to_visit = deque([base_url])
        self.all_text = ""
        self.page_count = 0
        
        # Parse base domain to stay within the same site
        self.base_domain = urlparse(base_url).netloc
        
    def is_valid_url(self, url):
        """Check if URL is valid for crawling"""
        parsed = urlparse(url)
        
        # Must be same domain
        if parsed.netloc != self.base_domain:
            return False
            
        # Skip common non-content files
        skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.ico']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Skip common non-content paths
        skip_paths = ['#', 'javascript:', 'mailto:', 'tel:']
        if any(url.lower().startswith(path) for path in skip_paths):
            return False
            
        return True
    
    def extract_links(self, soup, current_url):
        """Extract all valid links from the current page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            full_url = urljoin(current_url, href)
            
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        
        return links
    
    def extract_text_from_page(self, soup):
        """Extract readable text from a page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text from paragraphs, headings, and other content elements
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'article', 'section'])
        
        page_text = ""
        for element in text_elements:
            element_text = element.get_text(strip=True)
            if element_text and len(element_text) > 10:  # Filter out very short text
                page_text += element_text + "\n"
        
        return page_text
    
    def clean_text(self, text):
        """Remove problematic characters"""
        invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in invisible_chars:
            text = text.replace(char, '')
        return text
    
    def crawl(self):
        """Main crawling function"""
        print(f"Starting to crawl {self.base_url}")
        print(f"Max pages: {self.max_pages}, Delay: {self.delay}s")
        
        while self.to_visit and self.page_count < self.max_pages:
            current_url = self.to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            try:
                print(f"Crawling page {self.page_count + 1}: {current_url}")
                
                # Make request with headers and longer timeout
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                response = requests.get(current_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text from current page
                page_text = self.extract_text_from_page(soup)
                if page_text.strip():
                    self.all_text += f"\n\n--- PAGE: {current_url} ---\n\n"
                    self.all_text += page_text
                
                # Find new links to crawl
                new_links = self.extract_links(soup, current_url)
                for link in new_links:
                    if link not in self.visited_urls:
                        self.to_visit.append(link)
                
                # Mark as visited
                self.visited_urls.add(current_url)
                self.page_count += 1
                
                # Be respectful - add delay
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                continue
        
        print(f"Crawling completed. Visited {self.page_count} pages.")
        return self.all_text
    
    def save_to_file(self, filename="crawled_content.txt"):
        """Save all extracted text to file"""
        cleaned_text = self.clean_text(self.all_text)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"Content saved to {filename}")
            print(f"Total characters: {len(cleaned_text)}")
        except Exception as e:
            print(f"Error saving file: {e}")

# ------------- USAGE EXAMPLES

def crawl_jw_org():
    """Crawl JW.org website"""
    # Start with the main page
    base_url = "https://wol.jw.org/en/wol/"
    
    # Create crawler instance
    crawler = WebCrawler(
        base_url=base_url,
        max_pages=50,  # Adjust based on your needs
        delay=2  # Be respectful - 2 second delay between requests
    )
    
    # Start crawling
    all_text = crawler.crawl()
    
    # Save to file
    crawler.save_to_file("jw_org_content.txt")
    
    return all_text

def crawl_specific_section():
    """Crawl a specific section of JW.org"""
    # Example: Watchtower articles
    base_url = "https://wol.jw.org/en/wol/library/r1/lp-e/all-publications/watchtower"
    
    crawler = WebCrawler(
        base_url=base_url,
        max_pages=25,
        delay=1.5
    )
    
    all_text = crawler.crawl()
    crawler.save_to_file("watchtower_content.txt")
    
    return all_text

# ------------- TESTING AND DEBUGGING FUNCTIONS

def test_connection():
    """Test if we can connect to the website first"""
    test_url = "https://wol.jw.org/en/wol/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Testing connection to {test_url}...")
    
    try:
        response = requests.get(test_url, headers=headers, timeout=30)
        print(f"Success! Status code: {response.status_code}")
        print(f"Content length: {len(response.content)} bytes")
        
        # Test parsing
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        print(f"Page title: {title.get_text() if title else 'No title found'}")
        
        # Count links
        links = soup.find_all('a', href=True)
        print(f"Found {len(links)} links on the page")
        
        return True
        
    except requests.exceptions.Timeout:
        print("Connection timed out - try increasing timeout or check internet connection")
        return False
    except requests.exceptions.ConnectionError:
        print("Connection error - check internet connection")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False

def crawl_with_retry():
    """Crawl with automatic retry and better error handling"""
    # First test the connection
    if not test_connection():
        print("Connection test failed. Please check your internet connection.")
        return
    
    print("\nConnection test passed! Starting crawl...")
    
    base_url = "https://wol.jw.org/en/wol/"
    
    # Create crawler with more conservative settings
    crawler = RobustCrawler(
        base_url=base_url,
        max_pages=10,  # Start small
        delay=3,       # Longer delay
        max_retries=3  # Add retry capability
    )
    
    all_text = crawler.crawl()
    crawler.save_to_file("jw_org_content.txt")
    
    return all_text

class RobustCrawler(WebCrawler):
    """Enhanced crawler with retry logic and better error handling"""
    
    def __init__(self, base_url, max_pages=100, delay=1, max_retries=3):
        super().__init__(base_url, max_pages, delay)
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def make_request_with_retry(self, url):
        """Make request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{self.max_retries}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        
        return None
    
    def crawl(self):
        """Enhanced crawl with retry logic"""
        print(f"Starting robust crawl of {self.base_url}")
        print(f"Max pages: {self.max_pages}, Delay: {self.delay}s, Max retries: {self.max_retries}")
        
        while self.to_visit and self.page_count < self.max_pages:
            current_url = self.to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            print(f"\nCrawling page {self.page_count + 1}: {current_url}")
            
            # Make request with retry
            response = self.make_request_with_retry(current_url)
            
            if response is None:
                print(f"Failed to crawl {current_url} after {self.max_retries} attempts")
                continue
            
            try:
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text from current page
                page_text = self.extract_text_from_page(soup)
                if page_text.strip():
                    self.all_text += f"\n\n--- PAGE: {current_url} ---\n\n"
                    self.all_text += page_text
                    print(f"Extracted {len(page_text)} characters")
                
                # Find new links to crawl
                new_links = self.extract_links(soup, current_url)
                for link in new_links[:5]:  # Limit to first 5 links per page
                    if link not in self.visited_urls:
                        self.to_visit.append(link)
                
                print(f"Found {len(new_links)} new links")
                
                # Mark as visited
                self.visited_urls.add(current_url)
                self.page_count += 1
                
                # Be respectful - add delay
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"Error processing {current_url}: {e}")
                continue
        
        print(f"\nCrawling completed! Visited {self.page_count} pages.")
        print(f"Total content length: {len(self.all_text)} characters")
        return self.all_text

if __name__ == "__main__":
    # Option 1: Crawl main JW.org site
    crawl_jw_org()
    
    # Option 2: Crawl specific section (uncomment to use)
    # crawl_specific_section()
    
    # Option 3: Custom crawling
    # custom_url = "https://wol.jw.org/en/wol/library/r1/lp-e"
    # crawler = WebCrawler(custom_url, max_pages=30, delay=1)
    # text = crawler.crawl()
    # crawler.save_to_file("custom_crawl.txt")

# ------------- ADVANCED FEATURES

class AdvancedCrawler(WebCrawler):
    """Extended crawler with additional features"""
    
    def __init__(self, base_url, max_pages=100, delay=1, content_filters=None):
        super().__init__(base_url, max_pages, delay)
        self.content_filters = content_filters or []
        self.page_data = []  # Store individual page data
    
    def should_crawl_url(self, url):
        """Additional filtering for URLs"""
        # Add custom logic here
        # Example: Only crawl URLs containing certain keywords
        if self.content_filters:
            return any(filter_word in url.lower() for filter_word in self.content_filters)
        return True
    
    def extract_text_from_page(self, soup):
        """Enhanced text extraction"""
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        
        # Focus on main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # Fallback to all text
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up extra whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

# Example usage of advanced crawler
# advanced_crawler = AdvancedCrawler(
#     "https://wol.jw.org/en/wol/",
#     max_pages=20,
#     delay=2,
#     content_filters=['watchtower', 'awake', 'bible']  # Only crawl pages with these terms
# )
# advanced_text = advanced_crawler.crawl()
# advanced_crawler.save_to_file("filtered_content.txt")