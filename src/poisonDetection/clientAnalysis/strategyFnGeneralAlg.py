import copy

import numpy as np

from src.poisonDetection.clusteringHDBSCAN import run_hdbscan_clustering_algorithm, get_most_dense_cluster_label, \
    create_input_feature_list_per_fixed_round
from src.poisonDetection.tsneVisualisation import get_tsne_data_from_input_features, visualise_tsne_clusters_with_idx, \
    visualise_clusters_with_tsne
from src.poisonDetection.xaiMetrics import get_shap_feature_list_all


def client_analysis_fn_general_alg(weights_results, results, client_updates_list, server_round, **kwargs):
    round_idx = kwargs.get('round_idx')
    start_feature_idx = kwargs.get('start_feature_idx')
    target_label = kwargs.get('target_label')
    total_labels_per_client = kwargs.get('total_labels_per_client')
    sample_ct = kwargs.get('sample_count_for_plot')
    if(target_label!=None):
        end_feature_idx = start_feature_idx + total_labels_per_client
    else:
        end_feature_idx = start_feature_idx + sample_ct*total_labels_per_client

    min_cluster_size = kwargs.get('min_cluster_size')
    perplexity = kwargs.get('perplexity')
    show_poison_detection_graphs = kwargs.get('show_poison_detection_graphs')
    malicious_start_idx = kwargs.get('malicious_start_idx')
    malicious_end_idx = kwargs.get('malicious_end_idx')
    debug_info = kwargs.get('debug_info')
    is_eliminating_clients = kwargs.get('is_eliminating_clients')
    epsilon = kwargs.get('epsilon')
    if epsilon == None:
        epsilon = 0.01

    # debug_info.append(weights_results)
    # debug_info.append(results)
    debug_info.append(client_updates_list)

    # print('here')
    target_clients = client_updates_list[0].keys()
    client_ct = len(target_clients)

    # calculate shap feature list
    shap_feature_ori_list_all = get_shap_feature_list_all(weights_results, results, client_updates_list, **kwargs)
    # preprocess input features from shap original list
    shap_feature_per_client = create_input_feature_list_per_fixed_round(round_idx=round_idx,
                                                                        shap_feature_list=shap_feature_ori_list_all,
                                                                        total_labels_per_client=total_labels_per_client,
                                                                        client_ct=client_ct,
                                                                        start_feature_idx=start_feature_idx,
                                                                        end_feature_idx=end_feature_idx,
                                                                        target_clients=target_clients)
    # using hdbscan clustering from the preprocessed feature list per client
    hdbscan_labels, hdbscan_clusterer, colors = run_hdbscan_clustering_algorithm(
        input_feature_list=shap_feature_per_client, min_cluster_size=min_cluster_size, epsilon=epsilon)
    # tsne_data = get_tsne_data_from_input_features(shap_feature_per_client, perplexity=perplexity)
    # most_dense_cluster_label = get_most_dense_cluster_label(hdbscan_clusterer, hdbscan_labels)
    # is_target_min_label_index = hdbscan_labels == most_dense_cluster_label
    # malicious_list = [list(tsne_data[i]) for i in range(len(hdbscan_labels)) if is_target_min_label_index[i]]

    # display the poison detection with graphs
    if show_poison_detection_graphs:
        visualise_clusters_with_tsne(input_feature_list=shap_feature_per_client, label_list=hdbscan_labels,
                                     label_colors=colors, perplexity=perplexity,
                                     show_malicious_items=False, malicious_start_idx=None,
                                     malicious_end_idx=None, show_labels=True)

        # visualise_tsne_clusters_with_idx(tsne_data, hdbscan_labels, colors, np.array(malicious_list))
        # visualise_clusters_with_tsne(input_feature_list=shap_feature_per_client, label_list=hdbscan_labels,
        #                              label_colors=colors, perplexity=perplexity, show_malicious_items=True,
        #                              malicious_start_idx=malicious_start_idx,
        #                              malicious_end_idx=malicious_end_idx)
    # Running elimination algorithm

    # Removing elements from the list using the indices
    # weights_results_poison_updated = copy.deepcopy(weights_results)
    # results_poison_updated = copy.deepcopy(results)

    # Sort the indices in descending order to avoid index shifting issues
    if (target_label != None):
        poison_client_ids = general_algorithm_main_calc(client_updates_list, total_labels_per_client, hdbscan_labels)
    else:
        poison_client_ids = general_algorithm_main_calc(client_updates_list, sample_ct*total_labels_per_client,
                                                        hdbscan_labels)

    print('poison client ids: ', poison_client_ids)

    weights_results_poison_updated = copy.deepcopy(weights_results)
    results_poison_updated = copy.deepcopy(results)

    eliminated_ids = poison_client_ids.copy()
    poison_client_ids.sort(reverse=True) # method to remove the last items first so that indexes won't change
    eliminated_clients = []

    print(len(weights_results_poison_updated))
    if is_eliminating_clients:
        print('before update')
        for index in poison_client_ids:
            print('removing poison client at position: ', index)
            results_poison_updated.pop(index)
            eliminated_cil = weights_results_poison_updated.pop(index)
            eliminated_clients.append(eliminated_cil)
        print(len(weights_results_poison_updated))
        print('after update')
    else:
        print('Suspicious client elimination is not done!')

    return weights_results_poison_updated, results_poison_updated, eliminated_clients, eliminated_ids


'''main algorithm for new general poisoning detection'''
def general_algorithm_main_calc(client_updates_list, total_labels_per_client, hdbscan_labels):
    target_clients = list(client_updates_list[0].keys())
    # print(target_clients)
    # get all cluster ids, ignore -1 cluster id
    all_clusters_ids = np.unique(hdbscan_labels)
    # if np.any(all_clusters_ids == -1):
    #     all_clusters_ids = all_clusters_ids[all_clusters_ids != -1]

    # create an empty dict of arrays to get the positions for each features to be calculated for suspicious counts
    feature_positions = {}
    for i in all_clusters_ids:
        feature_positions[i] = []

    for i in range(len(hdbscan_labels)):
        if hdbscan_labels[i] in feature_positions.keys():
            feature_positions[hdbscan_labels[i]].append(i)
    print(feature_positions)
    diff_idxes_all = []

    # Main algorithm to detect poisoners: compare feature repetitions within the same cluster.
    # If different features are present, possible poisoning alert
    for i in all_clusters_ids:
        print(i)
        # List of numbers
        numbers = feature_positions[i]

        # Find the remainder when each number is divided by total output features/labels per client
        remainders = [num % total_labels_per_client for num in numbers]

        # Check if all the remainders are the same
        if all(remainder == remainders[0] for remainder in remainders):
            print("All cluster features are the same:", i)
        else:
            # print("Not all features are the same. Possible poisoning")
            # Find and isolate the numbers with different remainders
            different_idxes = [num for num, remainder in zip(numbers, remainders) if remainder != remainders[0]]
            # print("Cluster with different features:", different_idxes)
            # print(numbers)
            diff_idxes_all.extend(numbers)

    sus_ct = {}

    for i in list(target_clients):
        sus_ct[i] = 0

    # add a suspicious score for each client
    for i in diff_idxes_all:
        sus_client_position = i // (total_labels_per_client)
        sus_client = target_clients[sus_client_position]
        sus_ct[sus_client] += 1

    print(sus_ct)  # this is what we want!!

    # detecting poison clients
    poison_clients = []
    for key, value in sus_ct.items():
        if value >= total_labels_per_client / 2:
            poison_clients.append(key)  # CONVERTING TO AN INTEGER CAN BE A POTENTIAL BUG - yes it is, so eliminated!!!

    poison_idxes = []
    idxes_to_remove = list(client_updates_list[0].keys())
    for i in poison_clients:
        poison_idxes.append(idxes_to_remove.index(i))

    print('detected: ', poison_idxes)
    debugging_enabled = False
    poison_idx_viewing = True
    if poison_idx_viewing:
        # debugging operation (should update)
        my_list = list(client_updates_list[0].keys())
        # Values to find
        # values_to_find = ['1', '2', '3','4','5','6','7','8','9','10']
        values_to_find = ['1', '2', '3', '4', '5']

        # Find the indexes of the values in the list
        indexes = [i for i, value in enumerate(my_list) if value in values_to_find]
        print('original: ', indexes)
        if debugging_enabled:
            return indexes
    return poison_idxes


def client_analysis_fn_general_alg_debug(shap_feature_ori_list_all, client_updates_list, server_round, **kwargs):
  round_idx = kwargs.get('round_idx')
  start_feature_idx = kwargs.get('start_feature_idx')
  total_labels_per_client = kwargs.get('total_labels_per_client')
  end_feature_idx = start_feature_idx + total_labels_per_client

  min_cluster_size = kwargs.get('min_cluster_size')
  perplexity = kwargs.get('perplexity')
  show_poison_detection_graphs = kwargs.get('show_poison_detection_graphs')
  malicious_start_idx = kwargs.get('malicious_start_idx')
  malicious_end_idx = kwargs.get('malicious_end_idx')
  debug_info = kwargs.get('debug_info')

  debug_info.append(client_updates_list)

  target_clients = client_updates_list[0].keys()
  client_ct = len(target_clients)

  # preprocess input features from shap original list
  shap_feature_per_client = create_input_feature_list_per_fixed_round(round_idx=round_idx, shap_feature_list=shap_feature_ori_list_all,
                                                                      total_labels_per_client=total_labels_per_client, client_ct=client_ct,
                                                                      start_feature_idx=start_feature_idx, end_feature_idx=end_feature_idx,
                                                                      target_clients=target_clients)

  # using hdbscan clustering from the preprocessed feature list per client
  hdbscan_labels, hdbscan_clusterer, colors = run_hdbscan_clustering_algorithm(input_feature_list=shap_feature_per_client, min_cluster_size=min_cluster_size)
  # tsne_data = get_tsne_data_from_input_features(shap_feature_per_client, perplexity=perplexity)
  # most_dense_cluster_label = get_most_dense_cluster_label(hdbscan_clusterer, hdbscan_labels)
  # is_target_min_label_index = hdbscan_labels==most_dense_cluster_label
  # malicious_list = [list(tsne_data[i]) for i in range(len(hdbscan_labels)) if is_target_min_label_index[i]]
  # suspect_client_labels_list = obtain_suspect_clients(total_labels_per_client, hdbscan_labels, most_dense_cluster_label)

  # display the poison detection with graphs
  if show_poison_detection_graphs:
    visualise_clusters_with_tsne(input_feature_list=shap_feature_per_client, label_list=hdbscan_labels, label_colors=colors, perplexity=perplexity,
                                    show_malicious_items=False, malicious_start_idx=None,
                                    malicious_end_idx=None, show_labels=True)

    visualise_clusters_with_tsne(input_feature_list=shap_feature_per_client, label_list=hdbscan_labels,
    label_colors=colors, perplexity=perplexity, show_malicious_items=True, malicious_start_idx=malicious_start_idx,
    malicious_end_idx=malicious_end_idx)
    # visualise_tsne_clusters_with_idx(tsne_data, hdbscan_labels, colors, np.array(malicious_list))
  # Sort the indices in descending order to avoid index shifting issues
  # poison_client_ids = general_algorithm_main_calc(client_updates_list, hdbscan_labels)
  return shap_feature_per_client, hdbscan_labels