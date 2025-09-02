import numpy as np

from src.dataset.datasetHandler import get_testloader
from src.poisonDetection.clusteringHDBSCAN import run_hdbscan_clustering_algorithm, get_most_dense_cluster_label, \
    create_input_feature_list_per_fixed_round
from src.poisonDetection.tsneVisualisation import get_tsne_data_from_input_features, visualise_clusters_with_tsne, \
    visualise_tsne_clusters_with_idx
from src.poisonDetection.xaiMetrics import get_feature_contributions


def client_analysis_strategy_fn_random_poison(weights_results, results, client_updates_list, **kwargs):
    # target_clients = kwargs.get('client_ids')
    explainer_type = kwargs.get('explainer_type')
    total_rounds = kwargs.get('total_rounds')
    sample_count_for_plot = kwargs.get('sample_count_for_plot')
    target_label = kwargs.get('target_label')
    is_pca = kwargs.get('is_pca')
    num_pca_features = kwargs.get('num_pca_features')

    round_idx = kwargs.get('round_idx')
    start_feature_idx = kwargs.get('start_feature_idx')
    total_labels_per_client = kwargs.get('total_labels_per_client')
    end_feature_idx = start_feature_idx + total_labels_per_client
    shuffle = kwargs.get('shuffle')
    if shuffle == None:
        shuffle = False

    min_cluster_size = kwargs.get('min_cluster_size')
    perplexity = kwargs.get('perplexity')
    show_poison_detection_graphs = kwargs.get('show_poison_detection_graphs')
    malicious_start_idx = kwargs.get('malicious_start_idx')
    malicious_end_idx = kwargs.get('malicious_end_idx')
    debug_info = kwargs.get('debug_info')

    debug_info.append(weights_results)
    debug_info.append(results)
    debug_info.append(client_updates_list)
    print('len client updates list')
    print(len(client_updates_list))
    # if target_clients is None:
    target_clients = client_updates_list[0].keys()
    print('client keys')
    print(target_clients)
    client_ct = len(target_clients)
    shap_feature_ori_list_all = []

    # obtain shap list of features
    testloader = get_testloader(len_test=128, batch_size=128, shuffle=shuffle)

    for i in target_clients:
        print('target client:', i)
        shap_features = get_feature_contributions(testloader=testloader, explainer_type=explainer_type, total_rounds=total_rounds,
                                                  target_client_id=i,
                                                  client_updates_list=client_updates_list,
                                                  sample_count_for_plot=sample_count_for_plot,
                                                  target_label=target_label, is_pca=is_pca,
                                                  num_pca_features=num_pca_features)
        shap_feature_ori_list_all.append(shap_features)
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
        input_feature_list=shap_feature_per_client, min_cluster_size=min_cluster_size)
    tsne_data = get_tsne_data_from_input_features(shap_feature_per_client, perplexity=perplexity)
    most_dense_cluster_label = get_most_dense_cluster_label(hdbscan_clusterer, hdbscan_labels)
    is_target_min_label_index = hdbscan_labels == most_dense_cluster_label
    malicious_list = [list(tsne_data[i]) for i in range(len(hdbscan_labels)) if is_target_min_label_index[i]]
    suspect_client_labels_list = obtain_suspect_clients(total_labels_per_client, hdbscan_labels,
                                                        most_dense_cluster_label)
    poison_client_ids = []
    for cid in suspect_client_labels_list:
        if len(suspect_client_labels_list[
                   cid]) > total_labels_per_client // 2:  # consider that the suspect clients should have a minimum
            # suspecting features as at least half the total features in the client
            poison_client_ids.append(cid)
    print(suspect_client_labels_list)
    print(poison_client_ids)

    # display the poison detection with graphs
    if show_poison_detection_graphs:
        visualise_clusters_with_tsne(input_feature_list=shap_feature_per_client, label_list=hdbscan_labels,
                                     label_colors=colors, perplexity=perplexity,
                                     show_malicious_items=False, malicious_start_idx=None,
                                     malicious_end_idx=None, show_labels=True)

        # visualise_tsne_clusters_with_idx(tsne_data, hdbscan_labels, colors, np.array(malicious_list))
        # visualise_tsne_clusters_with_idx(tsne_data, label_list, colors, np.array(malicious_list))

        # visualise_clusters_with_tsne(input_feature_list=shap_feature_per_client, label_list=hdbscan_labels,
        # label_colors=colors, perplexity=perplexity, show_malicious_items=True, malicious_start_idx=10,
        # malicious_end_idx=50)

    # Removing elements from the list using the indices
    # weights_results_poison_updated = copy.deepcopy(weights_results)
    # results_poison_updated = copy.deepcopy(results)
    weights_results_poison_updated = weights_results
    results_poison_updated = results

    # Sort the indices in descending order to avoid index shifting issues

    eliminated_ids = poison_client_ids.copy()
    poison_client_ids.sort(reverse=True)
    eliminated_clients = []

    print('before update')
    print(len(weights_results_poison_updated))
    for index in poison_client_ids:
        print('removing poison client at position: ', index)
        results_poison_updated.pop(index)
        eliminated_cil = weights_results_poison_updated.pop(index)
        eliminated_clients.append(eliminated_cil)
    print(len(weights_results_poison_updated))
    print('after update')

    return weights_results_poison_updated, results_poison_updated, eliminated_clients, eliminated_ids


def obtain_suspect_clients(total_labels_per_client, cluster_label_list, suspect_cluster_label):
    suspect_clients = {}
    for i in range(len(cluster_label_list)):
        if cluster_label_list[i] == suspect_cluster_label:
            key = i // total_labels_per_client
            position = i % total_labels_per_client
            if key in suspect_clients:
                suspect_clients[key].append(position)
            else:
                suspect_clients[key] = [position]
    return suspect_clients
