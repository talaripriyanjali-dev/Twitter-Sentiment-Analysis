import pickle
import re
from html import escape, unescape

import nltk
import requests
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.tiekoetter.com",
    "https://nitter.space",
]


@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))


@st.cache_resource
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = " ".join(text)
    text = vectorizer.transform([text])

    sentiment = model.predict(text)
    return "Negative" if sentiment[0] == 0 else "Positive"


def clean_tweet_text(value):
    text = BeautifulSoup(value or "", "lxml").get_text(" ")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_tweets_from_nitter(username, limit=5):
    username = username.strip().lstrip("@")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", username):
        raise ValueError("Enter a valid Twitter username without @.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    session = requests.Session()
    session.trust_env = False
    errors = []

    for instance in NITTER_INSTANCES:
        tweets = try_nitter_rss(session, instance, username, headers, limit, errors)
        if tweets:
            return tweets

        tweets = try_nitter_profile(session, instance, username, headers, limit, errors)
        if tweets:
            return tweets

    error_preview = " | ".join(errors[:3])
    raise RuntimeError(
        "Could not fetch tweets from the available Nitter mirrors. "
        f"Details: {error_preview}"
    )


def try_nitter_rss(session, instance, username, headers, limit, errors):
    try:
        response = session.get(
            f"{instance}/{username}/rss",
            headers=headers,
            timeout=12,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        tweets = []

        for item in soup.find_all("item"):
            description = clean_tweet_text(item.description.text if item.description else "")
            title = clean_tweet_text(item.title.text if item.title else "")
            tweet_text = description or title

            if tweet_text:
                tweets.append({"text": tweet_text})

            if len(tweets) == limit:
                return tweets

        if not tweets:
            errors.append(f"{instance}: no tweets in RSS feed")
        return tweets
    except Exception as exc:
        errors.append(f"{instance}: RSS failed ({exc})")
        return []


def try_nitter_profile(session, instance, username, headers, limit, errors):
    try:
        response = session.get(
            f"{instance}/{username}",
            headers=headers,
            timeout=12,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        tweets = []

        for tweet in soup.select(".timeline-item .tweet-content"):
            tweet_text = clean_tweet_text(tweet.get_text(" "))
            if tweet_text:
                tweets.append({"text": tweet_text})

            if len(tweets) == limit:
                return tweets

        if not tweets:
            errors.append(f"{instance}: no tweets on profile page")
        return tweets
    except Exception as exc:
        errors.append(f"{instance}: profile failed ({exc})")
        return []


def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{escape(tweet_text)}</p>
    </div>
    """
    return card_html


def main():
    st.title("Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @)")

        if st.button("Fetch Tweets"):
            try:
                tweets_list = fetch_tweets_from_nitter(username, limit=5)
            except ValueError as e:
                st.warning(str(e))
                return
            except Exception as e:
                st.error(f"Error fetching tweets: {e}")
                return

            if len(tweets_list) == 0:
                st.warning("No tweets available.")
                return

            for tweet in tweets_list:
                tweet_text = tweet.get("text", "")

                if tweet_text.strip() == "":
                    continue

                sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                card_html = create_card(tweet_text, sentiment)
                st.markdown(card_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
