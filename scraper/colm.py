import openreview_scaper

if __name__ == "__main__":
    prefix = "colm"
    folder = f"../data/{prefix}/papers"
    
    openreview_scaper.scraper(prefix, folder)