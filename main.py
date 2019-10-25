from collections import Counter

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix


def get_training_and_evaluation_data():
    """
    Task 1
    """
    # Read in the file
    movie_data_xlsx = pd.read_excel("movie_reviews.xlsx")

    # Get reviews and sentiments where the data is marked as "train"
    reviews_training_data_list = movie_data_xlsx["Review"][movie_data_xlsx["Split"] == "train"]
    sentiment_training_data_list = movie_data_xlsx["Sentiment"][movie_data_xlsx["Split"] == "train"]

    # Get reviews and sentiments where the data is marked as "train"
    reviews_testing_data_list = movie_data_xlsx["Review"][movie_data_xlsx["Split"] == "test"]
    sentiment_testing_data_list = movie_data_xlsx["Sentiment"][movie_data_xlsx["Split"] == "test"]

    # Print out the gathered information
    print("There are: "
          "\n\t%d reviews used for training, where %d are positive, and %d are negative"
          "\n\t%d reviews used for testing, where %d are positive, and %d are negative"
          % (
              reviews_training_data_list.size,
              sentiment_training_data_list[sentiment_training_data_list == "positive"].size,
              sentiment_training_data_list[sentiment_training_data_list == "negative"].size,
              reviews_testing_data_list.size,
              sentiment_testing_data_list[sentiment_testing_data_list == "positive"].size,
              sentiment_testing_data_list[sentiment_testing_data_list == "negative"].size
          ))

    # Return the required data
    return reviews_training_data_list, sentiment_training_data_list, reviews_testing_data_list, sentiment_testing_data_list


def prepare_and_convert_training_data_to_word_list(reviews_training_data_list, minimum_word_length, minimum_number_of_word_occurrence):
    """
    Task 2

    This method does the following:
         1) Keep only alphanumeric and spaces between words
         2) Make them lowercase
         3) Split so it looks like this:
              [
                  1 [its, a, pretty, good, cast, but, the, film, h...
                  2 [this, ludicrous, film, offers, the, standard, ...
              ]
    :param reviews_training_data_list: reviews to go through
    :param minimum_word_length: minimum length of words to keep
    :param minimum_number_of_word_occurrence: minimum occurrence of words to keep
    :return: words above the minimum requirements
    """

    # Sanitise the data, make it lower case and finally split each row into words
    reviews_in_words = reviews_training_data_list.str \
        .replace("[^a-zA-Z ]", " ") \
        .str.lower() \
        .str.split()

    words_occurrences_key_value = {}  # This will store data like this: "entertaining:1421"
    for words in reviews_in_words:
        # Get words from each review
        for word in words:
            # For every word, do a check
            if len(word) >= minimum_word_length:
                if word in words_occurrences_key_value:
                    words_occurrences_key_value[word] += 1
                else:
                    words_occurrences_key_value[word] = 1

    words_meeting_criteria = []  # Words that meet our criteria, i.e., min length and occurrence frequency
    for word in words_occurrences_key_value:
        if words_occurrences_key_value[word] >= minimum_number_of_word_occurrence:
            words_meeting_criteria.append(word)

    # Return words above the minimum length and about the frequency
    return words_meeting_criteria


def frequency_of_words_in_feature(words_to_look_for, review_dataset_to_search):
    """
    Task 3

    This function goes through each word and sees how many reviews contain that word. If the word is found in the review, the counter is incremented

    :param words_to_look_for: words that we are trying to find, and will be keeping a tally of whether they show up in a review or not
    :param review_dataset_to_search: the reviews to look through
    :return: Frequency how many times the words show up in the reviews
    """
    # Sanitise the data, make it lower case and finally split each row into words
    sanitised_review_dataset_to_search = review_dataset_to_search.str \
        .replace("[^a-zA-Z0-9 ]", " ") \
        .str.lower() \
        .str.split()

    counter = Counter(words_to_look_for)
    for word in counter:
        counter[word] = 0

    for index, review in sanitised_review_dataset_to_search.iteritems():
        # Increment the counter for where the words match
        counter.update(set.intersection(set(review), words_to_look_for))

    return counter


def calculate_likelihood_using_laplace(frequency_of_positive_words, frequency_of_negative_words, total_number_of_positive_reviews, total_number_of_negative_reviews):
    """
    Task 4
    """
    # Each word extracted in test 2 -> binary feature of review, indicating that it's either present or absent
    # posterior 𝑃[𝜔𝑖 | x] is equal to likelihood 𝑃 [𝑥 | 𝜔i] and prior P[wi] divided by evidence P[x]

    likelihood_dictionary = {}  # Stores the frequencies of the words
    laplace_smoothing_alpha = 1
    for word in frequency_of_positive_words:
        # Go through each words, and apply laplace smoothing to them, using the len() of words as we need the total number distinct words
        likelihood_positive_laplace = np.divide(frequency_of_positive_words.get(word) + laplace_smoothing_alpha, total_number_of_positive_reviews + 2 * laplace_smoothing_alpha)
        likelihood_negative_laplace = np.divide(frequency_of_negative_words.get(word) + laplace_smoothing_alpha, total_number_of_negative_reviews + 2 * laplace_smoothing_alpha)

        # This will let us store the two values as a list for each key. eg: "happy : [0.05421, 0.0020212]"
        positive_negative_fractions_list = [likelihood_positive_laplace, likelihood_negative_laplace]
        likelihood_dictionary[word] = positive_negative_fractions_list

    # Calculate the priors
    positive_priors = np.divide(total_number_of_positive_reviews, total_number_of_positive_reviews + total_number_of_negative_reviews)
    negative_priors = np.divide(total_number_of_negative_reviews, total_number_of_negative_reviews + total_number_of_positive_reviews)

    return likelihood_dictionary, positive_priors, negative_priors


def predict_sentiment_label(review_text, positive_prior, negative_prior, likelihoods_of_word_dictionary):
    """
    Task 5

    :param review_text: Predict text of this sentiment
    :param positive_prior: The positive Prior
    :param negative_prior: The negative Prior
    :param likelihoods_of_word_dictionary: Dictionary with the likelihoods
    :return: positive or negative sentiment
    """
    review_text_as_words = review_text \
        .replace("[^a-zA-Z0-9 ]", " ") \
        .lower() \
        .split()

    positive_score = 0
    negative_score = 0
    for word in review_text_as_words:
        if word in likelihoods_of_word_dictionary:
            positive_score += np.math.log(likelihoods_of_word_dictionary[word][0] * positive_prior / (
                    likelihoods_of_word_dictionary[word][0] + likelihoods_of_word_dictionary[word][1]))
            negative_score += np.math.log(likelihoods_of_word_dictionary[word][1] * negative_prior / (
                    likelihoods_of_word_dictionary[word][0] + likelihoods_of_word_dictionary[word][1]))

    # TODO: ratios?
    if positive_score > negative_score:
        return "positive"
    else:
        return "negative"


def get_optimal_word_length_using_k_folds(reviews_training_data_list, sentiment_training_data_list, start_word_length_inclusive, end_word_length_inclusive):
    """
    Task 6

    Using k folds, it gets the best minimum word length

    :param reviews_training_data_list: Review data to train kfolds on
    :param sentiment_training_data_list: Sentiment data to train kfolds on
    :param start_word_length_inclusive: Minimum word length to start on
    :param end_word_length_inclusive: Minimum word length to finish on
    :return: optimal minimum word length
    """
    average_all_word_lengths_results = []  # keeps average result for each minimum word length run
    for current_min_word_length in range(start_word_length_inclusive, end_word_length_inclusive + 1):  # +1 to have it inclusive
        current_word_length_results = []  # Keeps results for this word length

        kf = model_selection.KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(reviews_training_data_list, sentiment_training_data_list):
            # Get the correct data based on how the kfold decided to divide up the data
            reviews_training_train_fold_data = reviews_training_data_list.iloc[train_index]
            reviews_training_test_fold_data = reviews_training_data_list.iloc[test_index]
            sentiment_training_train_fold_data = sentiment_training_data_list.iloc[train_index]
            sentiment_training_test_fold_data = sentiment_training_data_list.iloc[test_index]

            # print("Starting to convert training data into individual words")
            reviews_training_data_words_list = prepare_and_convert_training_data_to_word_list(reviews_training_train_fold_data, current_min_word_length, 100)
            # print("Finished converting training data into individual words")

            # print("Starting to get frequency of words in positive/negative reviews")
            frequency_of_words_in_positive_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_train_fold_data[sentiment_training_train_fold_data == "positive"])
            frequency_of_words_in_negative_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_train_fold_data[sentiment_training_train_fold_data == "negative"])
            # print("Finished getting the frequency of words in positive/negative reviews")

            total_positive_reviews = reviews_training_train_fold_data[sentiment_training_train_fold_data == "positive"].size
            total_negative_reviews = reviews_training_train_fold_data[sentiment_training_train_fold_data == "negative"].size

            # print("Starting to get likelihood using laplace")
            likelihood_laplace_dictionary, positive_priors, negative_priors = calculate_likelihood_using_laplace(frequency_of_words_in_positive_reviews,
                                                                                                                 frequency_of_words_in_negative_reviews,
                                                                                                                 total_positive_reviews,
                                                                                                                 total_negative_reviews)
            # print("Finished getting the likelihood using laplace")

            predicted_sentiment_list = []  # Used for building the list which is later fed into the confusion matrix
            for review in reviews_training_test_fold_data:
                predicted_sentiment = predict_sentiment_label(review, positive_priors, negative_priors, likelihood_laplace_dictionary)
                predicted_sentiment_list.append(predicted_sentiment)

            true_negative, false_positive, false_negative, true_positive = confusion_matrix(sentiment_training_test_fold_data, predicted_sentiment_list).ravel()

            # Append the percentage at the end of each fold
            current_word_length_accuracy = np.divide(np.sum([true_positive, true_negative]), len(sentiment_training_test_fold_data))
            print("A fold with minimum word length", current_min_word_length, "scored", current_word_length_accuracy * 100, "%")
            current_word_length_results.append(current_word_length_accuracy)

        average_score_for_min_length = np.mean(current_word_length_results)  # Get average result for this word length
        print("For word length", current_min_word_length, "the average accuracy was", average_score_for_min_length * 100, "%")
        average_all_word_lengths_results.append(average_score_for_min_length)  # get average of each run

    best_min_word_length = average_all_word_lengths_results.index(max(average_all_word_lengths_results)) + 1  # Getting best word length, +1 to account for list starting at 0
    print("The average scores were as follows:", average_all_word_lengths_results)
    print("Best average was achieved with minimum word length: ", best_min_word_length)
    return best_min_word_length


def task_2_to_5(reviews_testing_data_list, reviews_training_data_list, sentiment_testing_data_list, sentiment_training_data_list):
    """
    Does everything from task 2 to 5, just to keep the main method clean :)
    """

    print("Starting to convert training data into individual words")
    reviews_training_data_words_list = prepare_and_convert_training_data_to_word_list(reviews_training_data_list, 4, 100)
    print("reviews_training_data_words_list", reviews_training_data_words_list)
    print("Finished converting training data into individual words")

    print("Starting to get frequency of words in positive/negative reviews")
    frequency_of_words_in_positive_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "positive"])
    frequency_of_words_in_negative_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "negative"])
    print("frequency_of_words_in_positive_reviews", frequency_of_words_in_positive_reviews)
    print("frequency_of_words_in_negative_reviews", frequency_of_words_in_negative_reviews)
    print("Finished getting the frequency of words in positive/negative reviews")
    total_positive_reviews = reviews_training_data_list[sentiment_training_data_list == "positive"].size
    total_negative_reviews = reviews_training_data_list[sentiment_training_data_list == "negative"].size
    print("Starting to get likelihood using laplace and the priors")
    likelihood_laplace_dictionary, positive_priors, negative_priors = calculate_likelihood_using_laplace(frequency_of_words_in_positive_reviews, frequency_of_words_in_negative_reviews,
                                                                                                         total_positive_reviews, total_negative_reviews)
    print("likelihood_laplace_dictionary", likelihood_laplace_dictionary)
    print("positive_priors", positive_priors)
    print("negative_priors", negative_priors)
    print("Finished getting the likelihood using laplace and the priors")

    predicted_sentiment_list = []  # List of predicted sentiments which is later fed into the confusion matrix
    for review in reviews_testing_data_list:
        predicted_sentiment = predict_sentiment_label(review, positive_priors, negative_priors, likelihood_laplace_dictionary)
        predicted_sentiment_list.append(predicted_sentiment)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(sentiment_testing_data_list, predicted_sentiment_list).ravel()
    print("true negative", true_negative)
    print("false positive", false_positive)
    print("false negative", false_negative)
    print("true positive", true_positive)


def main():
    print("Starting to fetch data from file")
    reviews_training_data_list, sentiment_training_data_list, reviews_testing_data_list, sentiment_testing_data_list = get_training_and_evaluation_data()
    print("Finished fetching data from file")

    print("##############################")
    print("#####   Do task 2 to 5   #####")
    print("##############################")
    task_2_to_5(reviews_testing_data_list, reviews_training_data_list, sentiment_testing_data_list, sentiment_training_data_list)

    print("##############################")
    print("##### Moving onto task 6 #####")
    print("##############################")

    print("Starting to get optimal word length using kfolds")
    optimal_word_length = get_optimal_word_length_using_k_folds(reviews_training_data_list, sentiment_training_data_list, 1, 10)
    print("Finished getting optimal word length using kfolds")
    print("Starting to convert training data into individual words")
    # Get reviews above the optimal minimum word length
    reviews_training_data_words_list = prepare_and_convert_training_data_to_word_list(reviews_training_data_list, optimal_word_length, 100)
    print("Finished converting training data into individual words")

    print("Starting to get frequency of words in positive/negative reviews")
    frequency_of_words_in_positive_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "positive"])
    frequency_of_words_in_negative_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "negative"])
    print("Finished getting the frequency of words in positive/negative reviews")

    total_positive_reviews = reviews_training_data_list[sentiment_training_data_list == "positive"].size
    total_negative_reviews = reviews_training_data_list[sentiment_training_data_list == "negative"].size

    print("Starting to get likelihood using laplace")
    likelihood_laplace_dictionary, positive_priors, negative_priors = calculate_likelihood_using_laplace(frequency_of_words_in_positive_reviews, frequency_of_words_in_negative_reviews,
                                                                                                         total_positive_reviews, total_negative_reviews)
    print("Finished getting the likelihood using laplace")

    predicted_sentiment_list = []
    for review in reviews_testing_data_list:
        predicted_sentiment = predict_sentiment_label(review, positive_priors, negative_priors, likelihood_laplace_dictionary)
        predicted_sentiment_list.append(predicted_sentiment)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(sentiment_testing_data_list, predicted_sentiment_list).ravel()
    print("true negative", true_negative)
    print("false positive", false_positive)
    print("false negative", false_negative)
    print("true positive", true_positive)

    accuracy = np.divide(np.sum([true_positive, true_negative]), len(sentiment_testing_data_list))
    print("Accuracy from optimal word length", optimal_word_length, "obtained from the k folds, is", accuracy * 100, "%")

    # Option to write your own reviews
    print("##############################")
    print("# Optional: Your own reviews #")
    print("##############################")
    while True:
        review = input("Enter your review: ")
        predicted_sentiment = predict_sentiment_label(review, positive_priors, negative_priors, likelihood_laplace_dictionary)
        print("Beep Boop, I predict your review as:", predicted_sentiment)


main()
