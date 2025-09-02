import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def get_tsne_data_from_input_features(input_feature_list, perplexity=90, n_components=2, random_state=40):
    data = np.array(input_feature_list)
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


def visualise_clusters_with_tsne(input_feature_list, label_list, label_colors, perplexity=9,
                                 show_malicious_items=False, malicious_start_idx=None,
                                 malicious_end_idx=None, is_tsne_available=False, tsne_data=None, show_labels=False):
    if not is_tsne_available:
        data = np.array(input_feature_list)
        tsne = TSNE(n_components=2, random_state=40, perplexity=perplexity)
        tsne_data = tsne.fit_transform(data)
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    plt.scatter(x, y, c=label_colors)
    if show_labels:
        for i, txt in enumerate(label_list):
            plt.text(x[i], y[i], txt, ha='center', va='bottom')

    if show_malicious_items:
        subset_indices = range(malicious_start_idx,
                               malicious_end_idx)  # Define the indices of the subset of poison data
        subset_data = tsne_data[subset_indices]  # Select the subset of points from tsne_data
        subset_labels = label_list[subset_indices]  # Corresponding labels for the subset
        plt.scatter(subset_data[:, 0], subset_data[:, 1], c='red')  # Plot the subset in red

    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()


def visualise_tsne_clusters_with_idx(tsne_data, label_list, label_colors, malicious_list):
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=label_colors)
    plt.scatter(malicious_list[:, 0], malicious_list[:, 1], c='red')  # Plot the subset in red
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
