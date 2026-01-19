from scrapers.mha.mha_scraper import fetch_mha_tenders


def run_scrape() -> None:
    print("\n Starting MHA Tender Scraper\n")
    fetch_mha_tenders()
    print("\n MHA Scraping Completed\n")


def main() -> None:
    run_scrape()


if __name__ == "__main__":
    main()
