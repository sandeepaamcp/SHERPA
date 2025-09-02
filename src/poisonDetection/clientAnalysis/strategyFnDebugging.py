from util.constants import NUM_CLIENTS


def client_analysis_strategy_fn_debugging(weights_results, results, client_updates_list, server_round, **kwargs):
    debug_info = kwargs.get('debug_info')
    debug_info.append(weights_results)
    debug_info.append(results)
    debug_info.append(client_updates_list)
    print('came here debug')
    poison_client_ids = kwargs.get('dummy_poison_ids')
    is_eliminating_clients = kwargs.get('is_eliminating_clients')
    key_list = list(client_updates_list[0].keys())
    print(poison_client_ids)
    print(key_list)
    print(is_eliminating_clients)
    del_idxes = []
    for i in poison_client_ids:
        del_idxes.append(key_list.index(i))
    print(del_idxes)
    # weights_results_poison_updated = copy.deepcopy(weights_results)
    # results_poison_updated = copy.deepcopy(results)
    # weights_results_poison_updated = weights_results
    # results_poison_updated = results
    weights_results_poison_updated = []
    results_poison_updated = []

    # Sort the indices in descending order to avoid index shifting issues
    eliminated_clients = []
    eliminated_ids = []

    if is_eliminating_clients:
        eliminated_ids = poison_client_ids.copy()
        poison_client_ids.sort(reverse=True)
        eliminated_clients = []

        print('before update')
        print(len(weights_results))
        print(NUM_CLIENTS)

        for i in range(NUM_CLIENTS):
            if i not in del_idxes:
                weights_results_poison_updated.append(weights_results[i])
                results_poison_updated.append(results[i])
            else:

                print('removing poison client at position: ', i)
                eliminated_clients.append(weights_results[i])
    else:
        weights_results_poison_updated = weights_results
    # for index in del_idxes:
    # print('removing poison client at position: ', index)
    # # results_poison_updated.pop(index)
    # eliminated_cil = weights_results_poison_updated.pop(index)
    # eliminated_clients.append(eliminated_cil)
    # print(len(weights_results_poison_updated))
    # print('after update')
    # if is_eliminating_clients:
    #     print('before update')
    #     for index in poison_client_ids:
    #         print('removing poison client at position: ', index)
    #         results_poison_updated.pop(index)
    #         eliminated_cil = weights_results_poison_updated.pop(index)
    #         eliminated_clients.append(eliminated_cil)
    #     print(len(weights_results_poison_updated))
    #     print('after update')
    # else:
    #     print('Suspicious client elimination is not done!')

    print('after update')
    return weights_results_poison_updated, results_poison_updated, eliminated_clients, eliminated_ids
