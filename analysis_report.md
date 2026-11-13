# Name: Korbin Awender
# Course: CSCI 130
# Project: Week 4 Sentiment Analyzer
# File: analysis_report.md
# Date: 11/13/2026

Part 1 - Data and Setup

    The datset contained 1,000 positive reviews. The classes were balanced, so there was no significant class imbalance. This means the classifier could ot achieve a high accuracy by predicting the mjority class every time.

Part 2 - Build the analyser

    Accuracy: 0.81

Confusion Matrix:

[[171 28] [48 153]]

The model sees a TF-IDF representation of text that records which words appear and how often they appear, but it does not preserve the order of those words.

Part 3 - Build a Challenge Set and Break it
    
    The most difficult categories for my analyzer were negation and sarcasm, both with an accuracy of 60%. Negation was especially challenging because the TF-IDF bag-of-words representation does not preserve word order. The model can see that words such as "not" and "good" are present in a review, but it does not understand that "not" changes the meaning of "good." As a result, reviews such as "I was not impressed by this movie" and "The ending was not satisfying" were incorrectly classified as positive. The model was not missing information because it made a mistake; it was missing information because the text representation discarded the relationship between words before the classifier ever saw them.

    Sarcasm was also difficult because sarcastic reviews often contain positive words while expressing a negative meaning. For example, phrases such as "What a masterpiece of boredom" contain words that are usually associated with positive sentiment. Since the model relies heavily on word frequency and does not understand tone or intent, it struggled to correctly identify sarcastic statements.

Part 4 - Analysis Report
   
    Overall Performance
        The sentiment analyzer achieved an accuracy of 0.81 on the held-out movie review test set. While this result suggests the model performs reasonably well on review data similar to its training set, the challenge set revealed several important weaknesses.

    Challenge Set Accuracy by Category:
        Negation: 60%
        Sarcasm: 60%
        Mixed Sentiment: 80%
        Intensity: 80%
        Domain Shift: 100%

    The challenge set accuracy varied between categories. This demonstrates that overall accuracy alone does not fully show how a model performs in real-world situations. Some forms of language were much easier for the classifier than others.

    Most Difficult Categories
        The most difficult categories for the model were negation and sarcasm, both with an accuracy of 60%.

        Negation caused problems because the classifier uses a TF-IDF bag-of-words representation. This representation records which words are present and how often they appear, but it does not preserve word order. As a result, the model struggled with reviews such as "I was not impressed by this movie" and "The ending was not satisfying." Although these reviews are clearly negative to a human reader, the model was more influenced by positive words than by the relationship between those words and the negation.

        Sarcasm was also difficult because sarcastic reviews often contain positive vocabulary while expressing negative opinions. Reviews such as "What a masterpiece of boredom" include words that normally appear in positive reviews. Since the classifier cannot understand tone, intent, or context, it sometimes interpreted sarcastic statements as positive.

    How I Would Improve Negation Handling
        To improve performance on negation, I would use bigrams in addition to individual words. With a bag-of-words model, the words "not" and "good" are treated as separate features. A bigram approach would allow the phrase "not good" to become its own feature. This would preserve some word-order information and help the model recognize when negation changes the meaning of a sentence.

    The tradeoff is that a bigram model creates many more features, which increases processing time and memory usage. However, the additional complexity would likely improve the model's ability to correctly classify reviews that contain negation.

    Deployment Decision
        Based on the results, I would be a bit cautious about deploying this sentiment analyzer in a real-world environment. The model achieved respectable overall accuracy and performed well on mixed sentiment, intensity, and domain-shift examples. However, the challenge set showed that it struggles with negation and sarcasm, two patterns that occur frequently in natural language. If a company used this analyzer to identify dissatisfied customers and route them to human support representatives, some sarcastic complaints could be incorrectly classified as positive and never receive attention. Similarly, reviews containing negation could be misunderstood because the model does not understand how words relate to one another within a sentence.