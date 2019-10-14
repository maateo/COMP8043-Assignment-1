import pandas as pd


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
        .replace("[^a-zA-Z ]", "") \
        .str.lower() \
        .str.split()

    word_featured_frequency_in_review = {word: 0 for word in words_to_look_for}  # If word is in a review, +1 the frequency. Initialise the dictionary with 0 for every word
    for review in sanitised_review_dataset_to_search:
        for word_to_look_for in words_to_look_for:
            if word_to_look_for in review:
                if word_to_look_for in word_featured_frequency_in_review:
                    word_featured_frequency_in_review[word_to_look_for] += 1

    return word_featured_frequency_in_review


def main():
    reviews_training_data_list, sentiment_training_data_list, reviews_testing_data_list, sentiment_testing_data_list = get_training_and_evaluation_data()

    reviews_training_data_words_list = prepare_and_convert_training_data_to_word_list(reviews_training_data_list, 5, 1000)

    frequency_of_words_in_positive_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "positive"])
    frequency_of_words_in_negative_reviews = frequency_of_words_in_feature(reviews_training_data_words_list, reviews_training_data_list[sentiment_training_data_list == "negative"])


main()
