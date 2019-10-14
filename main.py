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


def main():
    reviews_training_data_list, sentiment_training_data_list, reviews_testing_data_list, sentiment_testing_data_list = get_training_and_evaluation_data()


main()
