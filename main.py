import time
import numpy as np
import functions as f

file_pth = r'H:\magistro_studijos\magis\data_eyes'
# intervals = [[500, 1000], [750, 1250], [1000, 1500]]
connectivity_features_fun = [f.pli_features, f.plv_features, f.imag_part_coh_features, f.coherence_features, f.adj_features, f.svd_features]

# if __name__ == '__main__':
#     start_time_multi = time.time()
#     graph_summary_df = f.summarize_graphs_seq(file_pth=file_pth,
#                                               intervals=intervals,
#                                               connectivity_features_fun=connectivity_features_fun)
#     end_time_multi = time.time()
#     print(end_time_multi - start_time_multi)
#     print(graph_summary_df)


if __name__ == '__main__':
    start_time_multi = time.time()

    intervals = []
    for i in range(1, 11):
        intervals.append(f.generate_intervals(0, 9000, i*100))

    intervals = [i for sublist in intervals for i in sublist]

    graph_summary_df = f.summarize_graphs_in_parallel_2(file_pth=f'{file_pth}/eeg_eyes_numpy_arrays_2.npy',
                                                        intervals=intervals,
                                                        connectivity_features_fun=connectivity_features_fun)
    end_time_multi = time.time()
    print(end_time_multi - start_time_multi)
    print(graph_summary_df.head())
    graph_summary_df.to_csv(f'{file_pth}/non_overlapping_df.csv')

