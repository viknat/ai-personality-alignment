import transformer.get_data as get_data
from transformer.get_data import Platform
from data_pipeline.data_pipeline_utils_reddit import process_reddit_data

if __name__ == '__main__':
    ###### First need to login in terminal "huggingface-cli login" using "huggingface_token.txt" or your own token
    ###### Then, we need to manually download .torrent file from academic torrent (or we need to find a valid magnet link)

    # IMPORTANT:
    # 1. Currently we download all .zst files using .torrent, so do ensure we have sufficient space in the computer to
    # hold it; Or, smartly use pause_download(), resume_download(), torrent_file.stop_download() as explained
    # in https://github.com/iw4p/torrentp/tree/master
    # 2. Please modify the variables "path_in_repo" and "repo_id" if needed. The repo used are for testing purpose only.
    # 3. Please modify the paths used below

    # # Step 1: download .zst files using .torrent
    # # If we finally use https://academictorrents.com/details/30dee5f0406da7a353aff6a8caa2d54fd01f2ca1,
    # # the structure after downloading is <output_folder>/reddit/comments
    # process_reddit_data.download_from_huggingface(
    #     r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/test.torrent",
    #     r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/")

    # Step 2: read .zst files, clean data, transform into our format, and push to HF
    process_reddit_data.push_to_huggingface(
        r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/reddit/comments/",
        r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/reddit/clean_comments/",
        delete_after_pushing=False)

    # # Step 3: stream data from HF as before
    # prev_ptr = None
    # ds_start_ptr = None
    # num_samples_for_now = 0
    # num_samples_we_want_in_total = 100
    # num_samples_we_want_each_time = 20
    # while True:
    #     samples, curr_ptr, ds_start_ptr = get_data.get_data(Platform.REDDIT, num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
    #     print("-----------------")
    #     print(len(samples))
    #     print(samples[0])
    #     num_samples_for_now += len(samples)
    #     prev_ptr = curr_ptr
    #     # do something of training using samples
    #     if num_samples_for_now >= num_samples_we_want_in_total:
    #         break
