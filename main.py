import time
import numpy as np
import functions as f

file_pth = r'H:\magistro_studijos\magis\data_eyes'
lengths_list = [300, 500, 1000]
connectivity_features_fun = [f.pli_features, f.plv_features, f.imag_part_coh_features, f.coherence_features, f.adj_features, f.svd_features]

# if __name__ == '__main__':
#     start_time_multi = time.time()
#     graph_summary_df = f.summarize_graphs_seq(file_pth=file_pth,
#                                               intervals=intervals,
#                                               connectivity_features_fun=connectivity_features_fun)
#     end_time_multi = time.time()
#     print(end_time_multi - start_time_multi)
#     print(graph_summary_df)

# just testing commit;)
if __name__ == '__main__':
    start_time_multi = time.time()

    for length in lengths_list:
        intervals = f.generate_intervals(0, 9000, length)

        graph_summary_df = f.summarize_graphs_in_parallel_2(file_pth=f'{file_pth}/eeg_eyes_numpy_arrays_2.npy',
                                                            intervals=intervals,
                                                            connectivity_features_fun=connectivity_features_fun)
        end_time_multi = time.time()
        print(end_time_multi - start_time_multi)
        graph_summary_df.to_csv(f'{file_pth}/non_overlapping_{length}_df.csv')

