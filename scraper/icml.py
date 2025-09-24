import openreview_scaper

if __name__ == "__main__":
    prefix = "icml"
    folder = f"../data/{prefix}/papers"
    
    openreview_scaper.scraper(prefix, folder)