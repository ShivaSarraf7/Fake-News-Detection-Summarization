from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def rank_articles(query, articles):

    # Remove articles without description
    articles = [
        article for article in articles
        if article["description"]
    ]


    if len(articles) == 0:
        return []


    documents = []


    for article in articles:
        documents.append(
            article["title"] +
            " " +
            article["description"] +
            " " +
            article.get("content", "")
        )


    # Add user query
    documents.insert(0, query)


    # TF-IDF conversion
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )


    # Compare query with articles
    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]


    # Add score to articles
    for i, article in enumerate(articles):

        article["similarity"] = round(
            float(similarity_scores[i]),
            3
        )

    ranked_articles = [
    article for article in articles
    if article["similarity"] > 0.1
    ]
    # Sort highest similarity first
    ranked_articles = sorted(
        articles,
        key=lambda x: x["similarity"],
        reverse=True
    )


    return ranked_articles