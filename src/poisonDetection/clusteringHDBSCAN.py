import numpy as np
import seaborn as sns
import hdbscan


def create_input_feature_list_per_fixed_round(round_idx, shap_feature_list, total_labels_per_client, client_ct,
                                              start_feature_idx, end_feature_idx, target_clients):
    shap_feature_per_client = []
    # start_feature_idx = 0
    # end_feature_idx = 10
    total_selecting_features = end_feature_idx - start_feature_idx
    print('total selecting features: ',total_selecting_features)
    for i in range(len(target_clients)):
        # shap_feature_per_client.append(shap_feature_list[i][ round_idx*total_rounds + target_label_idx])

        for j in range(total_selecting_features):
            shap_feature_per_client.append(shap_feature_list[i][j + round_idx * total_labels_per_client]) # here
            # round_idx means shap round index. If shap is run more than 1 round, we can have more than 0 for round_idx
    return shap_feature_per_client


def run_hdbscan_clustering_algorithm(input_feature_list, min_cluster_size=2, approx_min_span_tree=True,
                                     gen_min_span_tree=True, epsilon=0.01):
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, approx_min_span_tree=approx_min_span_tree,
                                        gen_min_span_tree=gen_min_span_tree,cluster_selection_epsilon=epsilon)
    hdbscan_labels = hdbscan_clusterer.fit_predict(input_feature_list)
    color_palette = sns.color_palette('pastel6', len(hdbscan_clusterer.labels_))
    cluster_colors = [color_palette[x] if x >= 0
                      else (0, 0, 0)
                      for x in hdbscan_labels]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbscan_clusterer.probabilities_)]

    return hdbscan_labels, hdbscan_clusterer, cluster_member_colors


def get_most_dense_cluster_label(hdbscan_clusterer, hdbscan_labels):
    # Get the probabilities of cluster membership
    cluster_probabilities = hdbscan_clusterer.probabilities_
    hdbscan_labels_no_outliers = hdbscan_labels[hdbscan_labels != -1]
    cluster_probabilities_no_outliers = cluster_probabilities[cluster_probabilities != 0]
    # Find the cluster label with the highest average probability
    most_dense_cluster_label = np.argmax(
        np.bincount(hdbscan_labels_no_outliers, weights=cluster_probabilities_no_outliers))
    return most_dense_cluster_label
