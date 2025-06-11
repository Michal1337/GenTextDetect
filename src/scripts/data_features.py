import os
import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Union

import nltk
import numpy as np
import pandas as pd
import spacy
import textstat
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.stats import entropy
from tqdm import tqdm

from params import (DATA_AI_PATH, DATA_HUMAN_PATH, FEATURES_PATH,
                    FEATURES_STATS_PATH)
from utils import get_csv_paths

# Download necessary NLTK data
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# Load SpaCy model for syntactic features
nlp = spacy.load("en_core_web_sm")

vader_analyzer = SentimentIntensityAnalyzer()


def d_metric(string: str) -> float:
    string_list = string.split()
    counts = np.unique(string_list, return_counts=True)[1]
    numerator = np.sum(counts * (counts - 1))
    n = len(string_list)
    if n < 2:
        return 0.0
    denominator = n * (n - 1)
    return numerator / denominator


def lexical_features(text: str) -> Dict[str, Union[int, float]]:
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    unique_words = set(words)
    return {
        "word_count": len(words),
        "character_count": sum(len(w) for w in words),
        "average_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "sentence_count": len(sentences),
        "TTR": len(unique_words) / len(words) if words else 0,
        "RTTR": np.sqrt(len(unique_words)) / len(words) if words else 0,
        "CTTR": len(unique_words) / ((len(words) * 2) ** 0.5) if words else 0,
        "DMetric": d_metric(text),
        "Mass": (
            (np.log10(len(words)) - np.log10(len(unique_words)))
            / (np.log10(len(words)) ** 2)
            if len(words) > 1
            else 0
        ),
        "stopword_ratio": (
            len([w for w in words if w.lower() in stop_words]) / len(words)
            if words
            else 0
        ),
    }


def nlp_features(text: str) -> dict:
    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc)
    entities = list(doc.ents)
    vader_scores = vader_analyzer.polarity_scores(text)

    return {
        "noun_ratio": pos_counts.get("NOUN", 0) / len(doc) if doc else 0,
        "verb_ratio": pos_counts.get("VERB", 0) / len(doc) if doc else 0,
        "adjective_ratio": pos_counts.get("ADJ", 0) / len(doc) if doc else 0,
        "average_sentence_length": (
            sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents))
            if list(doc.sents)
            else 0
        ),
        "std_sentence_length": (
            statistics.pstdev([len(sent.text.split()) for sent in doc.sents])
            if list(doc.sents)
            else 0
        ),
        "entity_count": len(entities),
        "syntactic_depth": max(
            (len(list(token.ancestors)) for token in doc), default=0
        ),
        "dependency_distance": (
            np.mean(
                [abs(token.head.i - token.i) for token in doc if token.head != token]
            )
            if doc
            else 0
        ),
        "sentiment": vader_scores["compound"] if vader_scores else 0,
    }


def readability_features(text: str) -> Dict[str, float]:
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog_index": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability": textstat.dale_chall_readability_score(text),
    }


def stylometric_features(text: str) -> Dict[str, Union[int, float]]:
    words = word_tokenize(text.lower())
    return {
        "punctuation_count": sum(1 for char in text if char in ".,;!?"),
        "entropy_score": entropy(list(Counter(words).values())),
    }


def discourse_features(text: str) -> Dict[str, Union[int, float]]:
    markers = {
        "however",
        "therefore",
        "moreover",
        "nevertheless",
        "thus",
        "on the other hand",
    }
    words = word_tokenize(text)
    count = sum(1 for w in words if w.lower() in markers)
    return {
        "discourse_marker_count": count,
        "discourse_marker_ratio": count / len(words) if words else 0,
    }


def repetition_features(text: str) -> Dict[str, float]:
    tokens = [w.lower() for w in word_tokenize(text)]

    def _ngram(n):
        ngrams = list(zip(*(tokens[i:] for i in range(n))))
        return 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0

    freq = Counter(tokens)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    return {
        "bigram_repetition_ratio": _ngram(2),
        "trigram_repetition_ratio": _ngram(3),
        "hapax_legomena_ratio": hapax / len(tokens) if tokens else 0,
    }


def syntactic_features(text: str) -> Dict[str, int]:
    return {
        "present_participle_count": sum(
            1 for w in word_tokenize(text) if w.lower().endswith("ing")
        ),
        "passive_voice_count": len(
            re.findall(
                r"\b(was|were|is|are|been|being)\s+\w+ed\b", text, flags=re.IGNORECASE
            )
        ),
    }


def cohesion_features(text: str) -> Dict[str, int]:
    text_l = text.lower()
    return {
        "conjunction_count": sum(
            text_l.count(w)
            for w in [" and ", " or ", " but ", " however ", " because ", " therefore "]
        ),
        "pronoun_count": sum(
            text_l.count(p)
            for p in [" i ", " you ", " he ", " she ", " they ", " we ", " it "]
        ),
        "contraction_count": sum(
            len(re.findall(p, text))
            for p in [
                r"\b\w+n't\b",
                r"\b\w+'re\b",
                r"\b\w+'ve\b",
                r"\b\w+'ll\b",
                r"\b\w+'d\b",
            ]
        ),
    }


def extract_features_single_text(text: str) -> Dict[str, Union[int, float]]:
    features = {}
    features.update(lexical_features(text))
    features.update(nlp_features(text))
    features.update(readability_features(text))
    features.update(stylometric_features(text))
    features.update(discourse_features(text))
    features.update(repetition_features(text))
    features.update(syntactic_features(text))
    features.update(cohesion_features(text))
    return features


def calc_features(texts: List[str], max_workers: int = 32) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        features_list = list(
            tqdm(executor.map(extract_features_single_text, texts), total=len(texts))
        )
    df = pd.DataFrame(features_list)
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
        print(f"Processing {path}...")
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
