import requests
from bs4 import BeautifulSoup
import time
import os
import re

# REQUIRED by Wikipedia: Set a proper User-Agent
HEADERS = {
    "User-Agent": "Assignment2/1.0 (skakumani@cpp.edu) Python/requests"
}

# Your 4 Wikipedia URLs
URLS = [
    "https://en.wikipedia.org/wiki/Avengers:_Endgame",
    "https://en.wikipedia.org/wiki/Real_Steel",
    "https://en.wikipedia.org/wiki/High_School_Musical",
    "https://en.wikipedia.org/wiki/War_(2019_film)",
    "https://en.wikipedia.org/wiki/Bugonia_(film)",
    "https://en.wikipedia.org/wiki/Alice_in_Wonderland_(1951_film)"
]

os.makedirs("documents", exist_ok=True)

def clean_text(text):
    # Fix missing spaces between words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Fix multiple spaces into one
    text = re.sub(r' +', ' ', text)
    # Fix multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_paragraph_text(content_div):
    paragraphs = content_div.find_all("p")
    para_parts = []
    for p in paragraphs:
        # Add a space between every tag's text to prevent word joining
        parts = []
        for element in p.descendants:
            if isinstance(element, str):
                parts.append(element)
        joined = " ".join(parts)
        cleaned = clean_text(joined)
        if cleaned:
            para_parts.append(cleaned)
    return "\n\n".join(para_parts)

def get_table_text(content_div):
    table_text = ""
    for table in content_div.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            # Use separator=" " so cell text doesn't get joined
            row_text = " | ".join(
                " ".join(cell.get_text(separator=" ", strip=True).split())
                for cell in cells
            )
            if row_text.strip():
                table_text += row_text + "\n"
    return table_text

def scrape_wikipedia_page(url, filename):
    # Polite delay between requests
    time.sleep(1)

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch {url} — Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract only the main article content
    content_div = soup.find("div", {"id": "mw-content-text"})

    if not content_div:
        print(f"Could not find content for {url}")
        return

    # Only remove junk elements
    for tag in content_div.find_all(["sup", "nav", "footer"]):
        tag.decompose()

    # Get paragraph and table text with spacing fixed
    para_text = get_paragraph_text(content_div)
    table_text = get_table_text(content_div)

    # Combine both
    full_text = para_text + "\n\n--- TABLE DATA ---\n\n" + table_text

    # Save to file
    with open(f"documents/{filename}", "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f" Saved: documents/{filename}")


#  Scrape all 4 pages
scrape_wikipedia_page(URLS[0], "avengers_endgame.txt")
scrape_wikipedia_page(URLS[1], "real_steel.txt")
scrape_wikipedia_page(URLS[2], "high_school_musical.txt")
scrape_wikipedia_page(URLS[3], "war_2019.txt")
scrape_wikipedia_page(URLS[4], "bugonia.txt")
scrape_wikipedia_page(URLS[5], "alice_in_wonderland.txt")

print("\n All pages scraped and saved successfully!")
