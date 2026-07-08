from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.text_rank import TextRankSummarizer

import nltk

nltk.download("punkt")


def summarize_text(text, sentences_count=3):

    if not text:
        return "No content available"


    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )


    summarizer = TextRankSummarizer()


    summary = summarizer(
        parser.document,
        sentences_count
    )


    result = []

    for sentence in summary:
        result.append(str(sentence))


    return " ".join(result)