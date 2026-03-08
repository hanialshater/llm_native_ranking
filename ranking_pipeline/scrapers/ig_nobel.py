"""Scraper for Ig Nobel Prize winners and their paper abstracts."""

import time
import requests
from bs4 import BeautifulSoup


IG_BASE = "https://improbable.com/ig/winners/"


def scrape_ig_nobel():
    """Scrape Ig Nobel winners list."""
    r = requests.get(IG_BASE, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    papers = []
    for entry in soup.find_all("div", class_="ig-winner"):
        title = entry.find("h3")
        year = entry.find("span", class_="year")
        category = entry.find("span", class_="category")
        doi = entry.find("a", href=lambda h: h and "doi.org" in h)

        papers.append({
            "title": title.text.strip() if title else "",
            "external_id": year.text.strip() if year else "",
            "url": doi["href"] if doi else "",
            "text": "",  # filled by fetch_abstract
            "category": category.text.strip() if category else "",
        })

    print(f"Found {len(papers)} Ig Nobel entries")
    return papers


def fetch_abstract(doi_url):
    """Fetch abstract via PubMed E-utilities."""
    if not doi_url:
        return ""

    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": doi, "retmode": "json"},
            timeout=15,
        )
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""

        summary = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ids[0], "retmode": "text", "rettype": "abstract"},
            timeout=15,
        )
        return summary.text.strip()
    except Exception as e:
        print(f"  Abstract fetch failed for {doi}: {e}")
        return ""


def scrape_all(n_papers=None, delay=1.0):
    """Scrape Ig Nobel winners and fetch abstracts."""
    papers = scrape_ig_nobel()
    if n_papers:
        papers = papers[:n_papers]

    for i, paper in enumerate(papers):
        if paper["url"]:
            print(f"  [{i+1}/{len(papers)}] Fetching abstract: {paper['title'][:60]}")
            paper["text"] = fetch_abstract(paper["url"])
            time.sleep(delay)

    with_text = [p for p in papers if p["text"]]
    print(f"Got abstracts for {len(with_text)}/{len(papers)} papers")
    return papers
