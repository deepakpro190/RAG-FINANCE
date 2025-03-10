import requests
from services.config import NEWS_API_KEY

class NewsAPIClient:
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key=NEWS_API_KEY):
        if not api_key:
            raise ValueError("API key for NewsAPI is missing. Please add it to the config file.")
        self.api_key = api_key

    def fetch_news(self, query, page=1, page_size=10):
        """
        Fetch news articles related to the query.

        Parameters:
        - query (str): The search term.
        - page (int): The page number for paginated results.
        - page_size (int): Number of articles per page.

        Returns:
        - list[dict]: List of news articles with titles, descriptions, and URLs.
        """
        params = {
            "q": query,
            "apiKey": self.api_key,
            "page": page,
            "pageSize": page_size,
            "language": "en",
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                return [
                    {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article.get("url"),
                    }
                    for article in data.get("articles", [])
                ]
            else:
                raise ValueError(f"Unexpected response: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return []

# Example usage
if __name__ == "__main__":
    client = NewsAPIClient()
    query = "Artificial Intelligence"
    articles = client.fetch_news(query)
    
    if articles:
        print(f"Found {len(articles)} articles for '{query}':")
        for article in articles:
            print(f"Title: {article['title']}")
            print(f"Description: {article['description']}")
            print(f"URL: {article['url']}\n")
    else:
        print("No articles found.")
