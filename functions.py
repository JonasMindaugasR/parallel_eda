# <<<<<<<<<<<<<<<<<<<<<<< Libraries >>>>>>>>>>>>>>>>>>>>>>>
import scipy.io
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy import signal
import numpy as np

import concurrent.futures
import multiprocessing

# <<<<<<<<<<<<<<<<<<<<<<< Load and prepare data data >>>>>>>>>>>>>>>>>>>>>>>

def load_eeg_data(file_path):
    # load data
    eeg_data = np.load(file_path, allow_pickle=True)

    # Initialize an empty dictionary
    eeg_data_dict = {}

    # Loop through the list of dictionaries
    for pair in eeg_data:
        # Extract the key and value from each dictionary
        for key, value in pair.items():
            # If the key is not in the dictionary, add it with an empty list
            if key not in eeg_data_dict:
                eeg_data_dict[key] = []

            # Append the value to the list associated with the key
            eeg_data_dict[key].append(value)

    return eeg_data_dict

def prepare_data(eeg_data_dict, epoch_start, epoch_end, label):
    # get eeg signals
    eeg_data_signals = np.stack(eeg_data_dict['result'], axis=0)

    # permute data and choose epoch
    eeg_data_permuted = np.transpose(eeg_data_signals[:, :, epoch_start:epoch_end], (0, 2, 1))

    # get label data
    eeg_label_data = np.array(eeg_data_dict['label'])

    # create two lists for each label
    output = list()


    for i, k in enumerate(eeg_label_data):
        if k == label:
            output.append(eeg_data_permuted[i, :, :])

    return output

# <<<<<<<<<<<<<<<<<<<<<<< Utility functions >>>>>>>>>>>>>>>>>>>>

def extract_features(features_fun, label):
    # Validate that features_fun is callable
    if not callable(features_fun):
        raise ValueError("features_fun must be a callable function.")

    # Extract features using list comprehensions
    label_corr = np.array(features_fun(label))

    return label_corr, features_fun.__name__

def create_intervals():

    return list


def generate_intervals(start, end, interval_length):
    """
    Generate a list of intervals from start to end, where each interval is of a fixed length.

    Parameters:
    - start (int): The starting point of the range (inclusive).
    - end (int): The ending point of the range (inclusive).
    - interval_length (int): The length of each interval.

    Returns:
    - intervals (list of lists): A list containing the start and end points of each interval.
    """
    intervals = []

    for i in range(start, end, interval_length):
        # Ensure that the end point of the last interval doesn't exceed the specified end
        interval_end = min(i + interval_length, end)
        intervals.append([i, interval_end])

    return intervals

# <<<<<<<<<<<<<<<<<<<<<<< Normalization >>>>>>>>>>>>>>>>>>>>>>>
# converting matrices to normalization
def _normalize_trial(trial):
    trial_avg = np.mean(trial)
    trial_std = np.std(trial)

    # Avoid division by zero by adding a small constant if trial_std is zero
    trial_std_safe = trial_std if trial_std != 0 else 1e-10

    trial = (trial - trial_avg) / trial_std_safe
    return trial

def min_max(trial):
    trial_min = np.min(trial)
    trial_max = np.max(trial)

    trial = (trial - trial_min) / (trial_max - trial_min)
    return trial

# <<<<<<<<<<<<<<<<<<<< Adjacency matrices >>>>>>>>>>>>>>>>>>>>
# --------------------Pearson's correlation--------------------
# using correlation
def adj_features(trial):
    feat = []
    trial_df = pd.DataFrame(trial, columns=list(range(1, 65)))
    corr_matrix = np.array(trial_df.corr())

    corr_matrix = _normalize_trial(corr_matrix)

    for i in range(np.shape(corr_matrix)[0]):
        feat.append(list(np.squeeze(corr_matrix[i, :])))

    return feat

# performing singular value decomposition on corr matrix
def svd_features(trial):
    feat = []
    trial_df = pd.DataFrame(trial, columns=list(range(1, 65)))
    corr_matrix = np.array(trial_df.corr())
    corr_matrix = _normalize_trial(corr_matrix)
    u, s, v = scipy.linalg.svd(corr_matrix)
    u = _normalize_trial(u)

    for i in range(np.shape(u)[0]):
        feat.append(list(np.squeeze(u[i, :])))

    # print(np.shape(feat))
    return feat

# --------------------Phase locking value--------------------
# phase locking value
def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(complex(0, 1) * (theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff)) / len(theta1)
    return plv

# phase locking value adjancency matrix
def plv_corr_matrix(trial, num_channels):
    corr_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            corr_matrix[i, j] = phase_locking_value(trial[:, i], trial[:, j])
            corr_matrix[j, i] = phase_locking_value(trial[:, i], trial[:, j])

    return (corr_matrix)

# extracting features from plv adjancency matrix
def plv_features(trial):
    feat = []
    num_channels = np.shape(trial)[1]
    corr_matrix = _normalize_trial(plv_corr_matrix(trial, num_channels))

    for i in range(np.shape(corr_matrix)[0]):
        feat.append(list(np.squeeze(corr_matrix[i, :])))
    return feat

# --------------------Phase lag index--------------------
# phase lag  index
def phase_lag_index(theta1, theta2):
    complex_phase_diff = np.sin(np.sign(theta1 - theta2))
    pli = np.abs(np.sum(complex_phase_diff)) / len(theta1)
    return pli

# phase locking index adjancency matrix
def pli_corr_matrix(trial, num_channels):
    corr_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            corr_matrix[i, j] = phase_lag_index(trial[:, i], trial[:, j])
            corr_matrix[j, i] = phase_lag_index(trial[:, i], trial[:, j])

    return (corr_matrix)

# extracting features from pli adjancency matrix
def pli_features(trial):
    feat = []
    num_channels = np.shape(trial)[1]
    corr_matrix = _normalize_trial(pli_corr_matrix(trial, num_channels))

    for i in range(np.shape(corr_matrix)[0]):
        feat.append(list(np.squeeze(corr_matrix[i, :])))
    return feat

# --------------------Coherence--------------------
# Coherence
def coherence(theta1, theta2, fs=120):
    _, Cxy = signal.coherence(theta1, theta2, fs=fs)
    coh = np.mean(Cxy)
    return coh

# Coherence adjancency matrix
def coherence_matrix(trial, num_channels):
    coh_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            coh_matrix[i, j] = coherence(trial[:, i], trial[:, j])
            coh_matrix[j, i] = coherence(trial[:, i], trial[:, j])

    return (coh_matrix)

# extracting features from Coherence adjancency matrix
def coherence_features(trial):
    feat = []
    num_channels = np.shape(trial)[1]
    coh_matrix = _normalize_trial(coherence_matrix(trial, num_channels))

    for i in range(np.shape(coh_matrix)[0]):
        feat.append(list(np.squeeze(coh_matrix[i, :])))
    return feat

# --------------------Imaginary part of coherence--------------------
# imaginary part of coherence
def imag_part_coherence(theta1, theta2, fs=120):
    # Calculate the cross-spectral density (CSD)
    _, Pxy = signal.csd(theta1, theta2, fs=fs)

    # Extract the imaginary part of the CSD
    imag_Pxy = np.imag(Pxy)
    coh = np.mean(imag_Pxy)
    return coh

# imaginary part of coherence adjancency matrix
def imag_part_coherence_matrix(trial, num_channels):
    imag_part_coh_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            imag_part_coh_matrix[i, j] = imag_part_coherence(trial[:, i], trial[:, j])
            imag_part_coh_matrix[j, i] = imag_part_coherence(trial[:, i], trial[:, j])

    return (imag_part_coh_matrix)

# extracting features from imaginary part of coherence adjancency matrix
def imag_part_coh_features(trial):
    feat = []
    num_channels = np.shape(trial)[1]
    imag_part_coh_matrix = _normalize_trial(imag_part_coherence_matrix(trial, num_channels))

    for i in range(np.shape(imag_part_coh_matrix)[0]):
        feat.append(list(np.squeeze(imag_part_coh_matrix[i, :])))
    return feat

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Graph metrics >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def calculate_graph_metrics(adj_matrix, threshold=0.5):
    # Create a graph from the correlation matrix (edges are correlations above a threshold)
    G = nx.Graph()

    # Add nodes
    num_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Add edges (only add if correlation is above a certain threshold)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid self-loops and double counting
            if abs(adj_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    # Calculate different graph metrics
    metrics = {}
    metrics['average_degree'] = sum(dict(G.degree()).values()) / num_nodes
    metrics['density'] = nx.density(G)
    metrics['avg_clustering'] = nx.average_clustering(G)
    metrics['diameter'] = nx.diameter(G) if nx.is_connected(G) else 0
    metrics['avg_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0

    return metrics

# calculate normalized difference
def summarize_graphs(list_of_graphs_1, list_of_graphs_2, feature, signal_length, epoch, graph_1_nm = "Graph 1", graph_2_nm = "Graph 2"):
    average_degree_1 = list()
    average_degree_2 = list()
    density_1 = list()
    density_2 = list()
    avg_clustering_1 = list()
    avg_clustering_2 = list()
    diameter_1 = list()
    diameter_2 = list()
    avg_shortest_path_1 = list()
    avg_shortest_path_2 = list()

    for num in range(100):
        average_degree_1.append(list_of_graphs_1[num]['average_degree'])
        average_degree_2.append(list_of_graphs_2[num]['average_degree'])
        density_1.append(list_of_graphs_1[num]['density'])
        density_2.append(list_of_graphs_2[num]['density'])
        avg_clustering_1.append(list_of_graphs_1[num]['avg_clustering'])
        avg_clustering_2.append(list_of_graphs_2[num]['avg_clustering'])
        diameter_1.append(list_of_graphs_1[num]['diameter'])
        diameter_2.append(list_of_graphs_2[num]['diameter'])
        avg_shortest_path_1.append(list_of_graphs_1[num]['avg_shortest_path'])
        avg_shortest_path_2.append(list_of_graphs_2[num]['avg_shortest_path'])

    df = pd.DataFrame(columns=['label', "feature", "signal_length", "epoch",
                               'average_degree', 'average_degree_std',
                               'density', 'density_std',
                               'avg_clustering', 'avg_clustering_std',
                               'diameter', 'diameter_std',
                               'avg_shortest_path', 'avg_shortest_path_std'])
    df.loc[0] = [graph_1_nm, feature, signal_length, epoch,
                 np.mean(average_degree_1), np.std(average_degree_1, ddof=1),
                 np.mean(density_1), np.std(density_1, ddof=1),
                 np.mean(avg_clustering_1), np.std(avg_clustering_1, ddof=1),
                 np.mean(diameter_1), np.std(diameter_1, ddof=1),
                 np.mean(avg_shortest_path_1), np.std(avg_shortest_path_1, ddof=1), ]
    df.loc[1] = [graph_2_nm, feature, signal_length, epoch,
                 np.mean(average_degree_2), np.std(average_degree_2, ddof=1),
                 np.mean(density_2), np.std(density_2, ddof=1),
                 np.mean(avg_clustering_2), np.std(avg_clustering_2, ddof=1),
                 np.mean(diameter_2), np.std(diameter_2, ddof=1),
                 np.mean(avg_shortest_path_2), np.std(avg_shortest_path_2, ddof=1), ]

    return df

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Non-parrallel functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def summarize_graphs_seq(file_pth, intervals, connectivity_features_fun):
    #load data
    data_dict = load_eeg_data(file_path = file_pth)

    graph_summary_results = []

    for epoch, interval in enumerate(intervals):
        # EEG signal preparation
        output1, output2 = prepare_data(data_dict, interval[0], interval[1])

        feature_summary_results = []

        for function in connectivity_features_fun:
            label_1_features, label_0_features = extract_features(function, output1, output2)

            label_1_metrics = list()
            label_0_metrics = list()

            for matrix in label_1_features:
                label_1_metrics.append(calculate_graph_metrics(matrix))

            for matrix in label_0_features:
                label_0_metrics.append(calculate_graph_metrics(matrix))

            summary_result = summarize_graphs(label_1_metrics, label_0_metrics,
                                                                feature = function.__name__,
                                                                signal_length = interval[1] - interval[0],
                                                                epoch = epoch,
                                                                graph_1_nm = "opened", graph_2_nm = "closed" )

            feature_summary_results.append(summary_result)

        df_summary_result = pd.concat(feature_summary_results, ignore_index=True)

        graph_summary_results.append(df_summary_result)

    graph_summary_df = pd.concat(graph_summary_results, ignore_index=True)

    return graph_summary_df

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Parrallel functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def summarize_graphs_in_parallel(file_pth, intervals, connectivity_features_fun):
    #load data
    data_dict = load_eeg_data(file_path = file_pth)

    graph_summary_results = []

    for epoch, interval in enumerate(intervals):
        # EEG signal preparation
        output1, output2 = prepare_data(data_dict, interval[0], interval[1])

        features_results = []

        # feature extraction for label_1 and label_0 preparation
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
            for function in connectivity_features_fun:
                feature_fun = [function] * len(output1)
                args = zip(feature_fun, output1, output2)
                result = pool.starmap(extract_features, args)
                features_results.extend(result)

        print(features_results)
        all_feature_results = []
        for res in features_results:
            all_feature_results.append(res)

        # Separate the outputs into two list
        # label_1_features, label_0_features, adj_features_nm = features_results

        print(len(features_results),type(features_results))

        # number of features
        num_features = 100

        label_1_features = calculate_graph_metrics(label_1_features)
        label_0_features = calculate_graph_metrics(label_0_features)

        # Summarize connectivity graphs
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
            summary_result = pool.starmap(summarize_graphs, zip(label_1_features, label_0_features,
                                                                adj_features_nm,
                                                                [interval[1] - interval[0]] * num_features,
                                                                [epoch] * num_features,
                                                                ["opened"] * num_features,
                                                                ["closed"] * num_features))

        df_summary_result = pd.concat(summary_result, ignore_index=True)
        graph_summary_results.append(df_summary_result)

    graph_summary_df = pd.concat(graph_summary_results, ignore_index=True)

    return graph_summary_df

def summarize_graphs_in_parallel_2(file_pth, intervals, connectivity_features_fun):
    #load data
    data_dict = load_eeg_data(file_path = file_pth)

    graph_summary_results = []

    # prepare data seperately - extract some interval
    for epoch, interval in enumerate(intervals):

        # EEG signal preparation
        output1 = prepare_data(data_dict, interval[0], interval[1], label=1)
        output2 = prepare_data(data_dict, interval[0], interval[1], label=0)

        features_results1 = []
        features_results2 = []

        # feature extraction for label_1 and label_0 preparation
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
            for function in connectivity_features_fun:
                feature_fun = [function] * len(output1)
                args1 = zip(feature_fun, output1)
                args2 = zip(feature_fun, output2)
                result1 = pool.starmap(extract_features, args1)
                result2 = pool.starmap(extract_features, args2)
                features_results1.extend(result1)
                features_results2.extend(result2)

        all_feature_results1 = []
        fun1 = []
        for res in features_results1:
            matrix, fun = res
            all_feature_results1.append(np.array(matrix))
            fun1.append(fun)

        all_feature_results2 = []
        fun2 = []
        for res in features_results2:
            matrix, fun = res
            all_feature_results2.append(np.array(matrix))
            fun2.append(fun)

        same_ordering = fun1 == fun2



        # number of features
        num_features = 100

        # with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        #     all_feature_results1 = pool.starmap(calculate_graph_metrics, all_feature_results1)
        #     all_feature_results2 = pool.starmap(calculate_graph_metrics, all_feature_results2)

        label_1_metrics = []
        label_0_metrics = []

        for matrix in all_feature_results1:
            label_1_metrics.append(calculate_graph_metrics(matrix))

        for matrix in all_feature_results2:
            label_0_metrics.append(calculate_graph_metrics(matrix))


        feature_summary_results = []

        function_set = list(dict.fromkeys(fun2))

        for num, fun in enumerate(function_set):
            summary_result = summarize_graphs(label_1_metrics[num*100:(num+1)*100],
                                              label_0_metrics[num*100:(num+1)*100],
                                                  feature=fun,
                                                  signal_length=interval[1] - interval[0],
                                                  epoch=epoch,
                                                  graph_1_nm="opened", graph_2_nm="closed")

            feature_summary_results.append(summary_result)

        df_summary_result = pd.concat(feature_summary_results, ignore_index=True)

        graph_summary_results.append(df_summary_result)

        print(f'epoch: {epoch}; interval: {interval}')

    graph_summary_df = pd.concat(graph_summary_results, ignore_index=True)

    return graph_summary_df
