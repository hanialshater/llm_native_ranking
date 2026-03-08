"""Scraper for Roger Ebert movie reviews."""

import time
import requests
from bs4 import BeautifulSoup


EBERT_BASE = "https://www.rogerebert.com/reviews"


def scrape_ebert_index(pages=50):
    """Scrape review index pages to get review URLs."""
    reviews = []
    for page in range(1, pages + 1):
        print(f"  Index page {page}/{pages}")
        try:
            r = requests.get(f"{EBERT_BASE}?page={page}", timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for card in soup.find_all("div", class_="review-card"):
                link = card.find("a")
                title = card.find("h3")
                stars = card.find("span", class_="star-rating")
                if link and link.get("href"):
                    href = link["href"]
                    if not href.startswith("http"):
                        href = f"https://www.rogerebert.com{href}"
                    reviews.append({
                        "url": href,
                        "title": title.text.strip() if title else "",
                        "stars": stars.text.strip() if stars else "",
                    })
        except Exception as e:
            print(f"    Failed page {page}: {e}")
        time.sleep(1.5)

    print(f"Found {len(reviews)} reviews")
    return reviews


def scrape_review(url):
    """Scrape a single review's full text."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    body = soup.find("div", class_="review-body") or soup.find("section", class_="review")
    return body.get_text(separator="\n").strip() if body else ""


def scrape_all(n_pages=5, n_reviews=None, delay=2.0):
    """Scrape Ebert reviews: index pages then full text."""
    index = scrape_ebert_index(pages=n_pages)
    if n_reviews:
        index = index[:n_reviews]

    results = []
    for i, review in enumerate(index):
        print(f"  [{i+1}/{len(index)}] {review['title'][:60]}")
        try:
            text = scrape_review(review["url"])
            if text:
                results.append({
                    "title": review["title"],
                    "url": review["url"],
                    "text": text,
                    "external_id": review.get("stars", ""),
                })
        except Exception as e:
            print(f"    Failed: {e}")
        time.sleep(delay)

    print(f"Scraped {len(results)} reviews")
    return results
