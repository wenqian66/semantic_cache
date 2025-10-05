import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

USE_SPACY = False
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    USE_SPACY = True
except (ImportError, OSError):
    nlp = None


def normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def extract_terms_spacy(text):
    doc = nlp(text.lower())
    terms = set()

    for token in doc:
        if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and
                not token.is_stop and
                len(token.lemma_) > 2 and
                token.lemma_.isalpha()):
            terms.add(token.lemma_)

    for i in range(len(doc) - 1):
        if (doc[i].pos_ in ['ADJ', 'NOUN'] and
                doc[i + 1].pos_ in ['NOUN', 'PROPN'] and
                not doc[i].is_stop and not doc[i + 1].is_stop):
            bigram = f"{doc[i].lemma_}_{doc[i + 1].lemma_}"
            terms.add(bigram)

    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT']:
            entity_term = re.sub(r'\s+', '_', ent.text.lower())
            terms.add(entity_term)

    return terms


def extract_terms_simple(text):
    text = normalize_text(text)
    words = text.split()
    terms = set()

    for word in words:
        if (word not in STOPWORDS and
            len(word) > 2 and
            word.isalpha()):
            stemmed = stemmer.stem(word)
            terms.add(stemmed)

    return terms


def extract_terms(text, use_spacy=USE_SPACY):
    if use_spacy and nlp is not None:
        return extract_terms_spacy(text)
    return extract_terms_simple(text)


def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

