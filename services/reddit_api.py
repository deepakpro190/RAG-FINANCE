import praw
from services.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

class RedditAPIClient:
    def __init__(self, client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def fetch_posts(self, subreddit, query, limit=10):
        """
        Fetch posts from a subreddit based on a search query.

        Parameters:
        - subreddit (str): The subreddit to search in.
        - query (str): The search term.
        - limit (int): Number of posts to retrieve.

        Returns:
        - list[dict]: List of posts with titles, URLs, and scores.
        """
        try:
            subreddit = self.reddit.subreddit(subreddit)
            posts = subreddit.search(query, limit=limit)
            return [
                {
                    "title": post.title,
                    "url": post.url,
                    "score": post.score
                }
                for post in posts
            ]
        except Exception as e:
            print(f"Error fetching Reddit posts: {e}")
            return []

# Example usage
if __name__ == "__main__":
    client = RedditAPIClient()
    posts = client.fetch_posts("stocks", "Artificial Intelligence", limit=5)
    for post in posts:
        print(f"Title: {post['title']}\nURL: {post['url']}\nScore: {post['score']}\n")
