import pandas as pd
import numpy as np


def get_training_and_evaluation_data():
    movie_review_xlsx = pd.read_excel("movie_reviews.xlsx")

    reviews_training_data_list = movie_review_xlsx["Review"][movie_review_xlsx["Split"] == "train"]
    sentiment_training_data_list = movie_review_xlsx["Sentiment"][movie_review_xlsx["Split"] == "train"]

    reviews_testing_data_list = movie_review_xlsx["Review"][movie_review_xlsx["Split"] == "test"]
    sentiment_testing_data_list = movie_review_xlsx["Sentiment"][movie_review_xlsx["Split"] == "test"]

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

    return reviews_training_data_list, sentiment_training_data_list, reviews_testing_data_list, sentiment_testing_data_list


def prepare_and_convert_training_data_to_word_list(reviews_training_data_list, minimum_word_length, minimum_number_of_word_occurrence):
    # This does the following:
    #  1) Keep only alphanumeric and spaces between words
    #  2) Make them lowercase
    #  3) Split so it looks like this:
    #       [
    #           1 [its, a, pretty, good, cast, but, the, film, h...
    #           2 [this, ludicrous, film, offers, the, standard, ...
    #       ]
    reviews_in_words = reviews_training_data_list.str \
        .replace("[^a-zA-Z ]", "") \
        .str.lower() \
        .str.split()

    words_occurrences_key_value = {}  # eg: "entertaining:1421"
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

    return words_meeting_criteria


def frequency_of_words_in_feature(words_to_look_for, review_dataset_to_search):
    """
    TODO: update this statement:
    # Function that goes through all reviews in review_dataset_to_search and counts for each of these words the number of reviews the word appears in

    :param words_to_look_for: words that we are trying to find, and will be keeping a tally of whether they show up in a review or not
    :param review_dataset_to_search: the reviews to look through
    :return:
    """
    sanitised_review_dataset_to_search = review_dataset_to_search.str \
        .replace("[^a-zA-Z0-9 ]", "") \
        .str.lower() \
        .str.split()

    word_featured_frequency_in_review = {word: 0 for word in words_to_look_for}  # If word is in a review, +1 the frequency. Initialise the dictionary with 0 for every word
    # for review in sanitised_review_dataset_to_search:
    #     for word_to_look_for in words_to_look_for:
    #         if word_to_look_for in review:
    #             word_featured_frequency_in_review[word_to_look_for] += 1

    for review in sanitised_review_dataset_to_search:
        unique_review_words = set(review)
        for unique_review_word in unique_review_words:
            if unique_review_word in words_to_look_for:
                word_featured_frequency_in_review[unique_review_word] += 1

    return word_featured_frequency_in_review


def calculate_likelihood_using_laplace(frequency_of_positive_words, frequency_of_negative_words, total_number_of_positive_reviews, total_number_of_negative_reviews):
    # Each word extracted in teast 2 -> binary heature of review, indicating that it's either present or abset
    # posterior 𝑃[𝜔𝑖 | x] is equal to likelihood 𝑃 [𝑥 | 𝜔i] and prior P[wi] divided by evidence P[x]

    frequencies_dictionary = {}
    for word in frequency_of_positive_words:
        likelihood_positive_laplace = np.divide(frequency_of_positive_words.get(word) + 1, total_number_of_positive_reviews + len(frequency_of_positive_words))
        likelihood_negative_laplace = np.divide(frequency_of_negative_words.get(word) + 1, total_number_of_negative_reviews + len(frequency_of_negative_words))

        pos_neg_fraction = [likelihood_positive_laplace, likelihood_negative_laplace]

        frequencies_dictionary[word] = pos_neg_fraction

    positive_priors = np.divide(total_number_of_positive_reviews, total_number_of_positive_reviews + total_number_of_negative_reviews)
    negative_priors = np.divide(total_number_of_negative_reviews, total_number_of_negative_reviews + total_number_of_positive_reviews)

    return frequencies_dictionary, positive_priors, negative_priors


def main():
    print("Starting to fetch data from file")
    reviews_training_data_list, sentiment_training_data_list, reviews_testing_data_list, sentiment_testing_data_list = get_training_and_evaluation_data()
    print("Finished fetching data from file")

    print("Starting to convert training data into individual words")
    reviews_training_data_words_list = prepare_and_convert_training_data_to_word_list(reviews_training_data_list, 12, 12)
    print("reviews_training_data_words_list", reviews_training_data_words_list)
    print("Finished converting training data into individual words")

    print("Starting to get frequency of words in positive/negative reviews")
    frequency_of_words_in_positive_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "positive"])
    frequency_of_words_in_negative_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "negative"])
    print("Finished getting the frequency of words in positive/negative reviews")

    total_positive_reviews = reviews_training_data_list[sentiment_training_data_list == "positive"].size
    total_negative_reviews = reviews_training_data_list[sentiment_training_data_list == "negative"].size

    print("Starting to get likelihood using laplace")
    likelihood_laplace_dictionary, positive_priors, negative_priors = calculate_likelihood_using_laplace(frequency_of_words_in_positive_reviews,
                                                                                                         frequency_of_words_in_negative_reviews,
                                                                                                         total_positive_reviews,
                                                                                                         total_negative_reviews)
    print("likelihood_laplace_dictionary", likelihood_laplace_dictionary)
    print("positive_priors", positive_priors)
    print("negative_priors", negative_priors)
    print("Finished getting the likelihood using laplace")


main()
