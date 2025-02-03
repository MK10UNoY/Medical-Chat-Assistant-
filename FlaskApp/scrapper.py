import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://medlineplus.gov/ency/article/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_FILE = "test_medical_data.json"

def is_valid_article(soup):
    """Checks if the scraped page is a valid medical article."""
    if soup.find("title") and "Page Not Found" in soup.find("title").text:
        return False
    return True  # If it has structured medical content, it's valid

def scrape_article(article_id):
    """Scrapes a single article given its six-digit ID."""
    url = f"{BASE_URL}{article_id}.htm"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Skipping {article_id}: Page not found")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    if not is_valid_article(soup):
        print(f"Skipping {article_id}: Invalid article")
        return None

    # Extract structured medical data
    data = {
        "id": article_id,
        "title": soup.find("h1").text.strip() if soup.find("h1") else "",
        "url": url,
        "summary": soup.select_one("#ency_summary p").text.strip() if soup.select_one("#ency_summary p") else "",
        "symptoms": [li.text.strip() for li in soup.select("ul li") if "symptom" in li.text.lower()],
        "causes": [p.text.strip() for p in soup.select("p") if "cause" in p.text.lower()],
        "treatments": [p.text.strip() for p in soup.select("p") if "treatment" in p.text.lower()]
    }
    
    return data

def main():
    all_data = []
    
    for i in range(1, 1000000):  # Run test from 000001 to 000100
        article_id = str(i).zfill(6)  # Convert number to six-digit format
        print(f"Scraping {article_id}...")

        try:
            data = scrape_article(article_id)
            if data:
                all_data.append(data)
        except Exception as e:
            print(f"Error scraping {article_id}: {e}")

        time.sleep(0.5)  # Avoid overloading the server

        # Save progress every 10 articles
        if i % 10 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_data, f, indent=4)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f, indent=4)

if __name__ == "__main__":
    main()
