from typing import List, Tuple

import numpy as np
from flwr.common import Metrics
import torch
from itertools import chain

from src.NN.NNConfig import get_nn
from util.constants import DEVICE


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def eval_exp(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.to(DEVICE)
    net.eval()
    real_all = []
    pred_all = []
    outputs_all = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            real_all.append(labels)
            pred_all.append(predicted)
            outputs_all.append(outputs.detach().tolist())
    loss /= len(testloader.dataset)
    accuracy = correct / total
    real_flat = [int(item) for tensor in real_all for item in tensor.view(-1)]
    pred_flat = [int(item) for tensor in pred_all for item in tensor.view(-1)]
    outputs_flat = list(chain(*outputs_all))
    return loss, accuracy, real_flat, pred_flat, outputs_flat


def get_mdl_from_weights(ori_weights):
  model = get_nn()
  weights = [torch.Tensor(weight) for weight in ori_weights]
  model_params = model.state_dict()
  for key, weight in zip(model_params.keys(), weights):
      model_params[key] = weight
  model.load_state_dict(model_params)
  return model

def get_weights_from_mdl(model):
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().numpy())  # Convert PyTorch tensors to numpy arrays
    return weights


def get_pred_from_models(round_no,testloader, target_client_list, client_updates_list):
  updates_list = client_updates_list[round_no]
  all_pred = []
  all_outputs = []
  for ref_cli in target_client_list:
    print('client ID: ',ref_cli)
    ref_mdl_weights = updates_list[ref_cli][0]
    mdl = get_mdl_from_weights(ref_mdl_weights)
    loss, acc, real, pred, per_cli_output = eval_exp(mdl,testloader)
    pred=np.array(pred)
    all_pred.append(pred)
    all_outputs.append(per_cli_output)
  return all_pred, all_outputs


def get_mdl_of_client_at_round(target_client_id, round_no, client_updates_list):
  # print(round_no)
  updates_list = client_updates_list[round_no]
  ref_mdl_weights = updates_list[target_client_id][0]
  mdl = get_mdl_from_weights(ref_mdl_weights)
  return mdl