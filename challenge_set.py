# Name: Korbin Awender
# Course: CSCI 130
# Project: Week 4 Sentiment Analyzer
# File: challenge_set.py
# Date: 11/13/2026

from sentiment_analyzer import (
    classifier,
    vectorizer,
    preprocess_text
)

challenge_reviews = [

    # 3.1 Negation
    ("The acting was not good at all.", "neg", "negation"),
    ("I was not impressed by this movie.", "neg", "negation"),
    ("This isn't a movie I would recommend.", "neg", "negation"),
    ("The plot was not interesting.", "neg", "negation"),
    ("The ending was not satisfying.", "neg", "negation"),

    # 3.2 Sarcasm
    ("Wonderful. Another three hours of my life wasted.", "neg", "sarcasm"),
    ("What a masterpiece of boredom.", "neg", "sarcasm"),
    ("I absolutely loved falling asleep halfway through.", "neg", "sarcasm"),
    ("A truly unforgettable experience for all the wrong reasons.", "neg", "sarcasm"),
    ("The writers clearly worked hard to make this movie this bad.", "neg", "sarcasm"),

    # 3.3 Mixed Sentiment
    ("The acting was excellent but the story was terrible.", "neg", "mixed"),
    ("Beautiful visuals but a boring plot.", "neg", "mixed"),
    ("The soundtrack was great although the characters were annoying.", "neg", "mixed"),
    ("Strong performances were wasted on weak writing.", "neg", "mixed"),
    ("The movie looked amazing, but it was difficult to finish.", "neg", "mixed"),

    # 3.4 Intensity
    ("The movie was fine.", "pos", "intensity"),
    ("The movie was pretty good.", "pos", "intensity"),
    ("The movie was great.", "pos", "intensity"),
    ("The movie was amazing.", "pos", "intensity"),
    ("The movie was extraordinary.", "pos", "intensity"),

    # 3.5 Domain Shift
    ("The restaurant had excellent food and great service.", "pos", "domain"),
    ("The burger was cold and disappointing.", "neg", "domain"),
    ("This laptop performs very well for the price.", "pos", "domain"),
    ("The headphones broke after two days.", "neg", "domain"),
    ("The hotel room was clean and comfortable.", "pos", "domain")
]

category_results = {}

print("\n===== CHALLENGE SET RESULTS =====\n")

for review, actual_label, category in challenge_reviews:

    processed = preprocess_text(review)

    vectorized = vectorizer.transform([processed])

    prediction = classifier.predict(vectorized)[0]

    print(f"Category: {category}")
    print(f"Review: {review}")
    print(f"Actual: {actual_label}")
    print(f"Predicted: {prediction}")
    print("-" * 60)

    if category not in category_results:
        category_results[category] = {
            "correct": 0,
            "total": 0
        }

    category_results[category]["total"] += 1

    if prediction == actual_label:
        category_results[category]["correct"] += 1

print("\n===== CATEGORY ACCURACY =====\n")

for category, stats in category_results.items():

    accuracy = (
        stats["correct"] /
        stats["total"]
    ) * 100

    print(f"{category}: {accuracy:.2f}%")