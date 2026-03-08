"""Scraper for Philosophize This! transcripts."""

import time
import requests
from bs4 import BeautifulSoup


BASE = "https://www.philosophizethis.org/transcripts"


def get_episode_urls():
    """Get all transcript URLs from the index page."""
    r = requests.get(BASE, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.find_all("a", href=lambda h: h and "/transcript/" in h)
    urls = []
    seen = set()
    for link in links:
        href = link["href"]
        if not href.startswith("http"):
            href = f"https://www.philosophizethis.org{href}"
        if href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def scrape_episode(url):
    """Scrape a single episode transcript."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Try multiple selectors for content
    content = (
        soup.find("div", class_="sqs-block-content")
        or soup.find("article")
        or soup.find("div", class_="entry-content")
    )
    title = soup.find("h1")
    title_text = title.text.strip() if title else url.split("/")[-1]
    text = content.get_text(separator="\n").strip() if content else ""

    # Extract episode number from URL or title
    external_id = ""
    for part in url.split("/")[-1].split("-"):
        if part.isdigit():
            external_id = part
            break

    return {
        "title": title_text,
        "url": url,
        "text": text,
        "external_id": external_id,
    }


def scrape_all(n_episodes=None, delay=2.0):
    """Scrape all (or n) episode transcripts."""
    urls = get_episode_urls()
    if n_episodes:
        urls = urls[:n_episodes]

    results = []
    for i, url in enumerate(urls):
        print(f"  [{i+1}/{len(urls)}] {url}")
        try:
            ep = scrape_episode(url)
            if ep["text"]:
                results.append(ep)
            else:
                print(f"    Warning: no text found")
        except Exception as e:
            print(f"    Failed: {e}")
        time.sleep(delay)

    print(f"Scraped {len(results)} episodes")
    return results
