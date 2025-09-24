import json
from tqdm import tqdm
import bibtexparser
from bs4 import BeautifulSoup
import requests
import time
import os
import ipdb
root = "https://papers.nips.cc"
base_url = "https://papers.nips.cc/paper_files/paper/{year}"
years = range(1987, 2025)

def collect_paper_urls(base_url, years, root):
    collected = []
    for year in tqdm(years):
        url = base_url.format(year=year)
        html = requests.get(url)
        if html.status_code != 200:
            print(f"Failed to retrieve data for year {year}")
            continue
        soup = BeautifulSoup(html.text, "html.parser")
        papers = soup.find_all("a", attrs={"title":"paper title"})
        for paper in papers:
            paper_url = paper.get("href")
            if paper_url is not None:
                collected.append(root + paper_url)
        time.sleep(1)
    return collected

def download_bib_and_abstract(urls, root):
    bibs = []
    abstracts = {}
    ok = True
    try:
        for url in tqdm(urls):
            # Download
            html = requests.get(url.strip())
            if html.status_code != 200:
                print(f"Failed to retrieve data for {url}, status code: {html.status_code}")
                continue

            # Get bibtex
            soup = BeautifulSoup(html.text, "html.parser")  
            bib_url = soup.find("a", string="Bibtex")
            if bib_url is None:
                print(f"No Bibtex found for {url}")
                continue
            bib_url = root + bib_url.get("href")
            bib_content = requests.get(bib_url.strip())
            if bib_content.status_code != 200:
                print(f"Failed to retrieve Bibtex for {url}")
                continue

            # Get abstract
            bibs.append(bib_content.text)
            bib_id = bib_content.text.split("\n")[0].split("{")[1].split(",")[0].strip()
            abstract_section = soup.find("h4", string="Abstract")
            if abstract_section is None:
                print(f"No abstract found for {url}")
                continue
            abstract_text = abstract_section.find_next("p").text.strip()
            abstracts[bib_id] = abstract_text
            time.sleep(1)
    except Exception as e:
        ok = False
        print(f"An error occurred: {e}")
    return bibs, abstracts, ok

class AddAbstract(bibtexparser.middlewares.BlockMiddleware):
    def __init__(self, abstract):
        self.abstract = abstract
        super().__init__()
    
    def transform_entry(self, entry, *args, **kwargs):              
        entry["abstract"] = self.abstract[entry.key].replace("{", " ").replace("}", " ")
        return entry
    
if __name__ == "__main__":
    # Collect URLs
    if os.path.exists("neurips_urls.txt"):
        with open("neurips_urls.txt", "r") as f:
            urls = list(f)
    else:
        urls = collect_paper_urls(base_url, years, root)
        with open("neurips_urls.txt", "a") as f:
            for url in urls:
                f.write(url + "\n")

    # Download bib files
    download = False
    if os.path.exists("neurips.bib"):
        with open("neurips.bib", "r") as f:
            bibs = f.readlines()
    else:
        download = True       

    if os.path.exists("neurips_abstracts.json"):
        with open("neurips_abstracts.json", "r") as f:
            abstracts = json.load(f)
    else:
        download = True
    
    if os.path.exists("neurips.err"):
        download = True

    if download:
        old_bibs = ""
        old_abstracts = {}
        if os.path.exists("neurips.err"):
            with open("neurips.err", "r") as f:
                n = int(f.read())
            with open("neurips.bib", "r") as f:
                old_bibs = f.read()
            with open("neurips_abstracts.json", "r") as f:
                old_abstracts = json.load(f)
            urls = urls[n:]
        bibs, abstracts, ok = download_bib_and_abstract(urls, root)
        
        bibs = old_bibs + "\n".join(bibs)
        abstracts = {**old_abstracts, **abstracts}
        with open("neurips.bib", "w") as f:
            f.write(bibs)
        with open("neurips_abstracts.json", "w") as f:
            json.dump(abstracts, f, indent=4)
        
        if not ok:
            print("An error occurred during the download process. Please check the logs.")
            with open("neurips.err", "w") as f:
                f.write(str(len(bibs)))
            exit(1) 
        else:
            try:
                os.remove("neurips.err")
            except FileNotFoundError:
                pass
    else:
        with open("neurips.bib", "r") as f:
            bibs = f.read()
        with open("neurips_abstracts.json", "r") as f:
            abstracts = json.load(f)

    library = bibtexparser.parse_string(bibs)
    library = bibtexparser.parse_string("", append_middleware=[AddAbstract(abstracts)], library=library)
    os.makedirs("../data/neurips", exist_ok=True)
    bibtexparser.write_file("../data/neurips/neurips.bib", library)
    