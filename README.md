# Twitter Sentiment Analysis

A Streamlit web app that predicts whether text or recent public tweets are positive or negative using a trained machine learning model.

## Features

- Analyze custom text entered by the user
- Fetch recent public tweets from a Twitter/X username
- Predict sentiment using a saved model and TF-IDF vectorizer
- Display tweet sentiment results in color-coded cards

## Project Structure

```text
.
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── Twitter_sentiment_Analysis.ipynb
└── training.1600000.processed.noemoticon.csv
```

## Requirements

- Python 3.10 or newer
- pip
- Internet access for fetching public tweets

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

After activating the virtual environment, run:

```bash
streamlit run app.py
```

Or run it directly with the project virtual environment on Windows:

```bash
venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

## Example Input

Positive example:

```text
I love this new phone, the camera is amazing and the battery lasts all day!
```

Negative example:

```text
This app keeps crashing and the support team never responds.
```

For tweet analysis, choose **Get tweets from user** and enter a username without `@`, for example:

```text
nasa
```

## Notes

- Tweet fetching uses public Nitter mirrors, so availability can change depending on network access and mirror uptime.
- If tweet fetching fails, manual text sentiment analysis will still work.
- `model.pkl` and `vectorizer.pkl` are required for prediction.
- The notebook contains the training workflow used to build the model.
