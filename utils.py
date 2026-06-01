import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


def preprocess(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = stemmer.stem(text)

    text = stopword.remove(text)

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text

def preprocess_sbert(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text