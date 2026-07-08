from summarizer.summarize import summarize_text


article = """
Salman Khan is an Indian actor and film producer.
He has appeared in many successful Bollywood films.
The actor recently announced his upcoming movie project.
Fans are excited about his new release.
The film is expected to release next year.
"""


summary = summarize_text(article, 2)


print("Summary:")
print(summary)