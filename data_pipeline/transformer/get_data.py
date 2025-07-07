from enum import Enum
import transformer.reddit_transformer_script as reddit
import transformer.twitter_transformer_script as twitter
import transformer.telegram_transformer_script as telegram


# Enum for the platform types
class Platform(Enum):
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    REDDIT = "reddit"


# The corresponding functions for each platform
def fetch_twitter_data(prev_ptr, ds_start_ptr, num_samples_we_want_each_time):
    samples, curr_ptr, ds_start_ptr = twitter.run(
        prev_ptr,
        ds_start_ptr,
        num_samples_we_want_each_time)
    return samples, curr_ptr, ds_start_ptr


def fetch_telegram_data(prev_ptr, ds_start_ptr, num_samples_we_want_each_time):
    samples, curr_ptr, ds_start_ptr  = telegram.run(
        num_samples_we_want_each_time,
        prev_ptr,
        ds_start_ptr)
    return samples, curr_ptr, ds_start_ptr


def fetch_reddit_data(prev_ptr, ds_start_ptr, num_samples_we_want_each_time):
    samples, curr_ptr, ds_start_ptr  = reddit.run(
               num_samples_we_want_each_time,
               prev_ptr,
               ds_start_ptr)
    return samples, curr_ptr, ds_start_ptr


# Get data with dataset splitting
def get_data(platform: Platform, max_num_samples_each_split, prev_ptr, ds_start_ptr):
    # Define a mapping of platforms to their corresponding functions
    platform_functions = {
        Platform.TWITTER: fetch_twitter_data,
        Platform.TELEGRAM: fetch_telegram_data,
        Platform.REDDIT: fetch_reddit_data
    }

    # Get the corresponding function for the platform
    platform_function = platform_functions.get(platform)

    if not platform_function:
        raise ValueError(f"Unsupported platform: {platform}")

    # Step 4: Perform dataset split and loop through the records
    samples, curr_ptr, ds_start_ptr = platform_function(prev_ptr, ds_start_ptr, max_num_samples_each_split)
    return samples, curr_ptr, ds_start_ptr

