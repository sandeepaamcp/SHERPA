import flwr as fl
import torch
torch.cuda.current_device()

from src.FLProcess.CustomFedAvg import CustomFedAvg
from src.FLProcess.FLUtil import weighted_average
from src.FLProcess.FlowerClient import FlowerClient
from src.NN.NNConfig import get_nn
from src.dataset.dataLoaderFactory import generate_data_loaders
from src.dataset.datasetStrategy import poison_strategy_with_non_iid_split, poison_strategy_for_multi_label_split
from src.poisonDetection.clientAnalysis.strategyFnGeneralAlg import client_analysis_fn_general_alg
from src.dataset.datasetHandler import non_iid_train_val_separate_split, get_classes
from src.poisonDetection.clientAnalysis.strategyFnRandomPoison import client_analysis_strategy_fn_random_poison
from util.constants import NUM_CLIENTS, DEVICE

# from sklearn.metrics import roc_curve, auc

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    kwargs_train = {'poison_type': 'target_poison', 'poison_ratio': 1, 'target_label': 1, 'target_clients': [1,2]}
    kwargs_val = {'poison_type': 'random_poison', 'poison_ratio': 0, 'target_clients': []}
    trainloaders, valloaders, testloaders = generate_data_loaders(kwargs_train, kwargs_val,
                          split_mechanism=non_iid_train_val_separate_split,
                          strategy=poison_strategy_with_non_iid_split,
                          len_train_data=1000, len_test_data=100,
                          random_ratio=1, is_visualize=False,
                          visualize_idx=0)

    client_updates_list = []
    aggregated_updates_list = []
    results = []
    weight_results = []
    eliminated_client_list = []
    eliminated_client_ids = []
    debug_info = []
    num_labels = len(get_classes())
    # should update these value based on the dataset: total_labels_per_client, target_label
    kwargs_poison = {'client_ids': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            'explainer_type': 'deep_exp', 'total_rounds': 1, 'sample_count_for_plot': 5,
                            'target_label': 1,
                            'is_pca': False, 'num_pca_features': 80,
                            'round_idx': 0, 'start_feature_idx': 0,
                            'total_labels_per_client': num_labels,
                            'min_cluster_size': 2, 'perplexity': 10,
                            'show_poison_detection_graphs': False, 'malicious_start_idx': 2,
                            'malicious_end_idx': 8,
                            'is_eliminating_clients': True,
                            'debug_info': debug_info}


    def client_fn(cid) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        net = get_nn().to(DEVICE)
        # trainloaders, valloaders, _ = get_train_val_test_loaders()
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(net, trainloader, valloader)

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        # min_fit_clients=10,
        # min_evaluate_clients=5,
        # min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        client_updates_list=client_updates_list,  # this contains updates from all the clients in all iterations,
        # however, in strategy_fn, only
        # current iteration updates are considered, which is later added to this list
        aggregated_updates_list=aggregated_updates_list,
        results_all=results,
        client_analysis_strategy_fn=client_analysis_fn_general_alg,
        strategy_kwargs=kwargs_poison,
        eliminated_client_list=eliminated_client_list,
        eliminated_client_ids=eliminated_client_ids,
        weight_results=weight_results
    )
    client_resources = None
    if DEVICE == "cuda":
        client_resources = {"num_cpus": 1, "num_gpus": 0.2}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        # client_manager = client_mnger,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"log_to_driver": False, "num_cpus": 5, "num_gpus": 1}
    )

