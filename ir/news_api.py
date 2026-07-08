import requests


API_KEY = "ce06f753e32e440d811613e8f3df2a56"


def get_related_news(query, page_size=10):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": f'"{query}"',
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)

    data = response.json()

    articles = []

    if data["status"] == "ok":

        for article in data["articles"]:

            if article["title"]:

                articles.append(
                    {
                        "title": article["title"] or "",
                        "description": article["description"] or "",
                        "content": article["content"] or "",
                        "url": article["url"]
                    }
                )

    return articles