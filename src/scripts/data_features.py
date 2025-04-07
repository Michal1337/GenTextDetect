import os
from collections import Counter
from typing import Callable, List, Union

import nltk
import numpy as np
import pandas as pd
import spacy
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.stats import entropy
from tqdm import tqdm

from params import (DATA_AI_PATH, DATA_HUMAN_PATH, FEATURES_PATH,
                    FEATURES_STATS_PATH)
from utils import get_csv_paths

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# Load SpaCy model for syntactic features
nlp = spacy.load("en_core_web_sm")


def lexical_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    unique_words = set(words)

    features = {
        "word_count": len(words),
        "character_count": sum(len(word) for word in words),
        "average_word_length": sum(len(word) for word in words) / len(words),
        "sentence_count": len(sentences),
        "unique_words_ratio": len(unique_words) / len(words),
        "stopword_ratio": len([word for word in words if word.lower() in stop_words])
        / len(words),
    }
    return features


def nlp_features(text):
    doc = nlp(text)
    pos_counts = Counter([token.pos_ for token in doc])
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment_scores = [token.sentiment for token in doc if token.sentiment != 0]

    features = {
        "noun_ratio": pos_counts.get("NOUN", 0) / len(doc),
        "verb_ratio": pos_counts.get("VERB", 0) / len(doc),
        "adjective_ratio": pos_counts.get("ADJ", 0) / len(doc),
        "average_sentence_length": sum(len(sent.text.split()) for sent in doc.sents)
        / len(list(doc.sents)),
        "entity_count": len(entities),
        "syntactic_depth": max([len(list(token.ancestors)) for token in doc] or [0]),
        "dependency_distance": np.mean(
            [abs(token.head.i - token.i) for token in doc if token.head != token]
        ),
        "average_sentiment_score": np.mean(sentiment_scores) if sentiment_scores else 0,
        "sentiment_variability": (
            np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
        ),
    }
    return features


def readability_features(text):
    features = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog_index": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability": textstat.dale_chall_readability_score(text),
    }
    return features


def stylometric_features(text):
    words = word_tokenize(text)
    bigrams = list(nltk.bigrams(words))
    trigrams = list(nltk.trigrams(words))

    features = {
        "bigram_count": len(bigrams),
        "trigram_count": len(trigrams),
        "punctuation_count": sum(1 for char in text if char in ".,;!?"),
        "entropy_score": entropy(list(Counter(words).values())),
    }
    return features


def discourse_features(text):
    discourse_markers = {
        "however",
        "therefore",
        "moreover",
        "nevertheless",
        "thus",
        "on the other hand",
    }
    words = word_tokenize(text)
    marker_count = sum(1 for word in words if word.lower() in discourse_markers)

    features = {
        "discourse_marker_count": marker_count,
        "discourse_marker_ratio": marker_count / len(words) if len(words) > 0 else 0,
    }
    return features


def extract_features_single_text(text):
    features = {}
    features.update(lexical_features(text))
    features.update(nlp_features(text))
    features.update(readability_features(text))
    features.update(stylometric_features(text))
    features.update(discourse_features(text))
    return features


def calc_features(texts: List[str]) -> pd.DataFrame:
    results = []
    for text in tqdm(texts):
        features = extract_features_single_text(text)
        results.append(features)

    df = pd.DataFrame(results)
    return df


def save_feature_stats(
    df: pd.DataFrame, stats: List[Union[str, Callable]], data_path: str, save_path: str
) -> None:
    df_stat = df.agg(stats).reset_index()

    data_name, model = data_path.split("/")[-1].split("_")
    model = model.removesuffix(".csv")

    df_stat["model"] = model
    df_stat["data"] = data_name
    df_stat.rename(columns={"index": "stat"}, inplace=True)
    df_stat.to_csv(
        save_path, mode="a", index=False, header=not pd.io.common.file_exists(save_path)
    )


def percentile(n: float) -> Callable:
    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = "percentile_{:02.0f}".format(n * 100)
    return percentile_


if __name__ == "__main__":
    STATS = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skew",
        "kurtosis",
        "var",
        percentile(0.1),
        percentile(0.2),
        percentile(0.3),
        percentile(0.4),
        percentile(0.5),
        percentile(0.6),
        percentile(0.7),
        percentile(0.8),
        percentile(0.9),
    ]
    paths = get_csv_paths(DATA_HUMAN_PATH) + get_csv_paths(DATA_AI_PATH, recursive=True)

    for path in paths:
        if path.split("_")[-1] == "human.csv":
            features_path = os.path.join(
                FEATURES_PATH,
                path.split("/")[-2],
                path.split("/")[-1].replace(".csv", "_features.csv"),
            )
        else:
            features_path = os.path.join(
                FEATURES_PATH,
                path.split("/")[-3],
                path.split("/")[-2],
                path.split("/")[-1].replace(".csv", "_features.csv"),
            )

        df = pd.read_csv(path)
        texts = df["text"].values
        df_features = calc_features(texts)
        df_features.to_csv(features_path, index=False)

        save_feature_stats(df_features, STATS, path, FEATURES_STATS_PATH)
