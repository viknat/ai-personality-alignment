from data_cleaner import fix_telegram_messages
import datetime
import transformer.reddit_transformer_script as reddit
import transformer.twitter_transformer_script as twitter
import transformer.telegram_transformer_script as telegram
import transformer.get_data as get_data
from transformer.get_data import Platform

if __name__ == '__main__':
    ###### first need to login in terminal "huggingface-cli login" using "huggingface_token.txt" or your own token

    # Explanations (for reddit and twitter):
    # num_samples_we_want_in_total: total number of samples we get
    # num_samples_we_want_each_time: the number of samples we get each time

    # Explanations (for telegram):
    # num_samples_we_want_in_total: total number of samples we get
    # num_samples_we_want_each_time: the number of lines of data we read each time (e.g. 5 lines of data get into one sample because of chunks)
    # Also remember to put a relatively large number but not a too large number for "num_samples_we_want_each_time",
    # because each time we load the whole channels and accounts dataset. A very small "num_samples_we_want_each_time"
    # leads to loading the two datasets too many times.

    prev_ptr = None
    ds_start_ptr = None
    num_samples_for_now = 0
    num_samples_we_want_in_total = 100
    num_samples_we_want_each_time = 20
    while True:
        samples, curr_ptr, ds_start_ptr = get_data.get_data(Platform.TWITTER, num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
        # samples, curr_ptr, ds_start_ptr = get_data.get_data(Platform.REDDIT, num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
        # samples, curr_ptr, ds_start_ptr = get_data.get_data(Platform.TELEGRAM, num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
        print("-----------------")
        print(len(samples))
        print(samples[0])
        num_samples_for_now += len(samples)
        prev_ptr = curr_ptr
        # do something of training using samples
        if num_samples_for_now >= num_samples_we_want_in_total:
            break