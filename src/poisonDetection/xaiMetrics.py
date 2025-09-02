import numpy as np
import pandas as pd
import shap
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from src.FLProcess.FLUtil import get_mdl_of_client_at_round
from src.dataset.datasetHandler import get_testloader
import util.constants as const


def get_shap_for_mdl(testloader, target_client_id, round_no, client_updates_list, sample_count_for_test=100,
                     sample_count_for_plot=3, target_label=None, show_plot=True,
                     explainer_type='deep_exp', is_zero_vector_items=False):
    # testloader = get_testloader(len_test=128, batch_size=128, shuffle=shuffle)
    batch = next(iter(testloader))
    images, actual_out = batch
    background = images[:sample_count_for_test]
    if target_label != None:
        target_idx = actual_out[sample_count_for_test:].tolist().index(target_label) + sample_count_for_test
        test_images = images[target_idx:target_idx + 1]
    else:
        test_images = images[sample_count_for_test:sample_count_for_test + sample_count_for_plot]
        if is_zero_vector_items:  # this is to test if we send test sample with 0 vector value for shap
            zero_tensors_list = [torch.zeros(test_images[0].shape) for _ in range(sample_count_for_plot)]
            test_images = torch.stack(zero_tensors_list)

    mdl_0 = get_mdl_of_client_at_round(target_client_id=target_client_id, round_no=round_no,
                                       client_updates_list=client_updates_list)
    output = mdl_0(test_images)
    probabilities = torch.softmax(output, dim=1).tolist()[0]

    if type(background) == list:
        if explainer_type == 'deep_exp':
            e = shap.DeepExplainer(mdl_0, background)
        elif explainer_type == 'grad_exp':
            e = shap.GradientExplainer(mdl_0, background)
        elif explainer_type == 'sampling_exp':
            e = shap.SamplingExplainer(mdl_0, background)
        else:
            raise ValueError('Invalid explainer type')

        shap_values = e.shap_values(test_images)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

        if show_plot:
            shap.image_plot(shap_numpy, -test_numpy, show=False)
            fig = plt.gcf()
            allaxes = fig.get_axes()
            for x in range(1, len(allaxes) - 1):
                allaxes[x].set_title('{:.2%}'.format(probabilities[x - 1]), fontsize=14)
            plt.show()

        return shap_numpy, test_numpy, probabilities
    elif type(background) != list:
        if explainer_type == 'deep_exp':
            e = shap.DeepExplainer(mdl_0, [background])
        elif explainer_type == 'grad_exp':
            e = shap.GradientExplainer(mdl_0, [background])
        elif explainer_type == 'sampling_exp':
            e = shap.SamplingExplainer(mdl_0, [background])
        else:
            raise ValueError('Invalid explainer type')

        shap_values = e.shap_values([test_images])
        return shap_values, test_images, probabilities
    else:
        raise ValueError('Dataset not found')


def get_shap_PCA_flattened_values(client_updates_list, target_label, target_round, target_clients, n_components=10,
                                  explainer_type='grad_exp', shuffle=False):
    shap_vals_all = []
    testloader = get_testloader(len_test=128, batch_size=128, shuffle=shuffle)
    for i in target_clients:
        print('xai on client ', i)
        shap_vals, test_vals, prob = get_shap_for_mdl(client_updates_list, target_client_id=i,
                                                      round_no=target_round,
                                                      target_label=target_label, show_plot=False,
                                                      explainer_type=explainer_type, testloader=testloader)
        # shap_vals_all.append(shap_vals)
        # shap_vals, test_vals, prob = get_shap_for_mdl(target_client_id=i,
        #                                               round_no=target_round, client_updates_list=client_updates_list,
        #                                               target_label = 1, show_plot=False, explainer_type = explainer_type)
        shap_vals_all.append(shap_vals)
    shap_vals_all_flattened = []
    for num in shap_vals_all:
        for pred in num:
            shap_vals_all_flattened.append(pred[0].flatten())
    pca = PCA(n_components=n_components)
    shap_pca = pca.fit_transform(shap_vals_all_flattened)
    return shap_pca


def get_tsne_plot_for_shap_vals(shap_pca, classes, repetitions_per_class=10):
    n = repetitions_per_class  # Number of repetitions for each element
    elements = classes  # Elements to repeat

    y_results = [x for x in elements for _ in range(n)]

    y_train = np.array(y_results)
    tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=8.0)  # Reduce to 2 dimensions for visualization
    z = tsne.fit_transform(shap_pca)
    df = pd.DataFrame()
    df["y"] = y_train
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df)

    # plt.title(" T-SNE projection")
    plt.legend(fontsize=10, loc='upper left')


def get_feature_contributions(testloader, explainer_type, total_rounds, target_client_id,
                              client_updates_list, sample_count_for_plot, target_label,
                              is_pca=True, num_pca_features=100):
    shap_vals_all = []
    prob_all = []

    for i in range(total_rounds):
        # print('running round:', i)
        shap_vals, test_vals, prob = get_shap_for_mdl(testloader=testloader, target_client_id=target_client_id,
                                                      round_no=i, client_updates_list=client_updates_list,
                                                      sample_count_for_plot=sample_count_for_plot,
                                                      target_label=target_label, explainer_type=explainer_type,
                                                      show_plot=False)
        shap_vals_all.append(shap_vals)
        prob_all.append(prob)
    shap_vals_all_flattened = []
    for num in shap_vals_all:
        for pred in num:
            for i in pred:
                shap_vals_all_flattened.append(i.flatten())
    if not is_pca:
        return shap_vals_all_flattened
    else:
        pca = PCA(n_components=num_pca_features)  # Reduce to 2 dimensions for visualization
        shap_pca = pca.fit_transform(shap_vals_all_flattened)
        return shap_pca


# util function to calculate the shap values list
def get_shap_feature_list_all(weights_results, results, client_updates_list, **kwargs):
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

    min_cluster_size = kwargs.get('min_cluster_size')
    perplexity = kwargs.get('perplexity')
    show_poison_detection_graphs = kwargs.get('show_poison_detection_graphs')
    malicious_start_idx = kwargs.get('malicious_start_idx')
    malicious_end_idx = kwargs.get('malicious_end_idx')
    debug_info = kwargs.get('debug_info')
    shuffle = kwargs.get('shuffle')
    if shuffle == None:
        shuffle = False

    debug_info.append(weights_results)
    debug_info.append(results)
    debug_info.append(client_updates_list)
    # print('len client updates list')
    # print(len(client_updates_list))
    # if target_clients is None:
    target_clients = client_updates_list[0].keys()
    client_ct = len(target_clients)
    shap_feature_ori_list_all = []

    # obtain shap list of features
    testloader = get_testloader(len_test=128, batch_size=128, shuffle=shuffle)
    for i in target_clients:
        # print('target client:', i)
        shap_features = get_feature_contributions(testloader=testloader, explainer_type=explainer_type, total_rounds=total_rounds,
                                                  target_client_id=i,
                                                  client_updates_list=client_updates_list,
                                                  sample_count_for_plot=sample_count_for_plot,
                                                  target_label=target_label, is_pca=is_pca,
                                                  num_pca_features=num_pca_features)
        shap_feature_ori_list_all.append(shap_features)

    return shap_feature_ori_list_all
