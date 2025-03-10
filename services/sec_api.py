import requests
from bs4 import BeautifulSoup

class SECFilingsClient:
    BASE_URL = "https://www.sec.gov/edgar/searchedgar/companysearch.html"

    def fetch_filings(self, cik, filing_type="10-K"):
        """
        Fetch SEC filings for a given company using its CIK.

        Parameters:
        - cik (str): The Central Index Key of the company.
        - filing_type (str): Type of filing (e.g., "10-K", "10-Q").

        Returns:
        - list[dict]: List of filings with filing type, description, and URL.
        """
        params = {
            "CIK": cik,
            "type": filing_type
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            filings = []

            for row in soup.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) > 3:
                    filings.append({
                        "type": cols[0].text.strip(),
                        "description": cols[1].text.strip(),
                        "url": cols[2].find("a")["href"]
                    })

            return filings
        except Exception as e:
            print(f"Error fetching SEC filings: {e}")
            return []

# Example usage
if __name__ == "__main__":
    client = SECFilingsClient()
    filings = client.fetch_filings("0000320193", filing_type="10-K")
    for filing in filings:
        print(f"Type: {filing['type']}\nDescription: {filing['description']}\nURL: {filing['url']}\n")
