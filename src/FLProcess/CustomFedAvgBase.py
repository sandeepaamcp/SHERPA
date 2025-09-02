from logging import WARNING, INFO, DEBUG
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
import time

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy


class CustomFedAvgBase(FedAvg):

    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            client_updates_list=[],
            aggregated_updates_list=[],
            results_all=[],
            client_analysis_strategy_fn=None,
            strategy_kwargs={},
            eliminated_client_list=[],
            eliminated_client_ids=[],
            weight_results=[]
    ) -> None:

        super().__init__()

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.client_updates_list = client_updates_list
        self.aggregated_updates_list = aggregated_updates_list
        self.results_all = results_all
        self.client_analysis_strategy_fn = client_analysis_strategy_fn
        self.strategy_kwargs = strategy_kwargs
        self.eliminated_client_list = eliminated_client_list
        self.eliminated_client_ids = eliminated_client_ids
        self.weight_results = weight_results

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        log(INFO, f"Attempting to initialize parameters")
        print('Attempting to initialize parameters')
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        start_time = time.time()

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # print(results)
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        iter_params_dict = {}
        for item, fit_res in results:
            iter_params_dict[item.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)

        self.results_all.append(results)
        # self.client_updates_list.append(weights_results)
        self.client_updates_list.append(iter_params_dict)
        self.weight_results.append(weights_results)

        # weights_results_poison_updated = copy.deepcopy(weights_results)
        # results_poison_updated = copy.deepcopy(results)

        if self.client_analysis_strategy_fn is not None:
            weights_results_poison_updated, results_poison_updated, eliminated_clients, eliminated_ids = self.client_analysis_strategy_fn(
                weights_results, results,
                [iter_params_dict], **self.strategy_kwargs)
            self.eliminated_client_ids.append(eliminated_ids)
            self.eliminated_client_list.append(eliminated_clients)
            # print('aggregating poison removed weights')
            aggregated_params_arr = aggregate(
                weights_results_poison_updated)  # aggregate on updated results with poisoners removed
        else:
            aggregated_params_arr = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_params_arr)
        self.aggregated_updates_list.append(aggregated_params_arr)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in
                           results_poison_updated]  # aggreate on updated results with poisoners removed
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            print("No fit_metrics_aggregation_fn provided")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time} seconds")

        return parameters_aggregated, metrics_aggregated

    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
    #     clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
    #
    #     configs = {'lr':0.01}
