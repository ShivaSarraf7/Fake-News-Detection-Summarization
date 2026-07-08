from ir.news_api import get_related_news
from ir.retrieve import rank_articles



query = "Salman Khan"


print("Fetching articles...")

articles = get_related_news(
    query,
    page_size=10
)


print("Ranking articles...")


results = rank_articles(
    query,
    articles
)



for article in results:

    print("\nTitle:")
    print(article["title"])

    print("Similarity:")
    print(article["similarity"])

    print("URL:")
    print(article["url"])

    print("---------------------")