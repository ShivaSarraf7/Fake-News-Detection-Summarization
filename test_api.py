from ir.news_api import get_related_news


results = get_related_news(
    "India economy"
)


for news in results:
    print(news["title"])
    print(news["url"])
    print("----------------")