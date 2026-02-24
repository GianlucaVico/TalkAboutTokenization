import openreview_scaper

if __name__ == "__main__":
    prefix = "neurips"
    folder = f"../data/{prefix}2025/papers"
    
    openreview_scaper.scraper(prefix, folder, year_filter=lambda y: y == "2025")