import random
from torch.utils.data import Subset

from src.dataset.poisoning import random_swap, random_poison, target_poison


def non_iid_split_strategy_fixed_label(dataset, client_dataset_len_list, num_clients, label_list, random_ratio,
                                       **kwargs):
    all_subsets = []
    for i in range(num_clients):
        target_label = label_list[i]  # Choose the label you want to create a subset for
        limit_per_client = client_dataset_len_list[i]
        random_count = int(limit_per_client * random_ratio)

        if (random_count > limit_per_client):
            print('random count is above 100%. Setting 100% randomness.')
            random_count = limit_per_client

        fixed_count = limit_per_client - random_count
        print('processing client: ', i)
        # subset_indices = [idx for idx, (_, label) in enumerate(dataset) if label == target_label][:fixed_count]
        subset_indices = []
        count = 0
        for idx, (_, label) in enumerate(dataset):
            if label == target_label:
                subset_indices.append(idx)
                count += 1
                if count == fixed_count:
                    break
        random_indices = random.sample(range(1, len(dataset)), random_count)

        subset_indices.extend(random_indices)
        random.shuffle(subset_indices)
        subset_dataset = Subset(dataset, subset_indices)
        all_subsets.append(subset_dataset)

    return all_subsets


def poison_strategy_with_non_iid_split(dataset, client_dataset_len_list, num_clients, label_list, random_ratio,
                                       **kwargs):
    all_subsets = []
    if len(label_list) < num_clients:
        print('Note: lesser labels for clients than number of requested clients, may consist of same label for '
              'multiple clients')

    for i in range(num_clients):
        # target_label = label_list[i]  # Choose the label you want to create a subset for
        limit_per_client = client_dataset_len_list[i]
        random_count = int(limit_per_client * random_ratio)

        if random_count > limit_per_client:
            print('random count is above 100%. Setting 100% randomness.')
            random_count = limit_per_client

        fixed_count = limit_per_client - random_count
        print('processing client: ', i)

        # subset_indices = [idx for idx, (_, label) in enumerate(dataset) if label == target_label][:fixed_count]
        # count = 0
        # for idx, (_, label) in enumerate(dataset):
        #     if label == target_label and count <= fixed_count:
        #         subset_indices.append(idx)
        #         count += 1
        #         #if count == fixed_count:
        #              #break
        if len(label_list) < num_clients:
            target_label = label_list[i % len(label_list)]  # since num_clients exceed the no. of labels,
            # multiple clients will get the same label
            print('assigning label ', target_label)
            all_indices = [idx for idx, (_, label) in enumerate(dataset) if label == target_label]
            if fixed_count * num_clients > len(
                    all_indices):  # check if all possible labels are not sufficient for all shared clients
                fixed_count = len(all_indices) // num_clients  # distribute data among all clients if not sufficient
            subset_indices = all_indices[i * fixed_count:(i + 1) * fixed_count]

        else:
            target_label = label_list[i]  # Choose the label you want to create a subset for
            all_indices = [idx for idx, (_, label) in enumerate(dataset) if label == target_label]
            if fixed_count * num_clients > len(
                    all_indices):  # check if all possible labels are not sufficient for all shared clients
                fixed_count = len(all_indices) // num_clients
            subset_indices = all_indices[:fixed_count]

        random_indices = random.sample(range(1, len(dataset)), random_count)

        subset_indices.extend(random_indices)
        random.shuffle(subset_indices)
        subset_dataset = Subset(dataset, subset_indices)
        target_clients = kwargs.get('target_clients')
        if target_clients is None:
            print('no poisoning for client: ', i)
            all_subsets.append(subset_dataset)

        elif i in target_clients:
            if kwargs.get('poison_type') == 'random_swap':
                print('perform random swap poisoning')
                subset_poisoned = random_swap(subset_dataset, kwargs.get('poison_ratio'))

            elif kwargs.get('poison_type') == 'random_poison':
                print('performing random poisoning of labels')
                subset_poisoned = random_poison(subset_dataset, kwargs.get('poison_ratio'), label_list)

            elif kwargs.get('poison_type') == 'target_poison':
                print('performing targeted poisoning of labels')
                subset_poisoned = target_poison(subset_dataset, kwargs.get('poison_ratio'), kwargs.get('target_label'))
            else:
                raise ValueError('Keyword argument for poison type is invalid.')

            all_subsets.append(subset_poisoned)

        else:
            print('no poisoning for client: ', i)
            all_subsets.append(subset_dataset)

    return all_subsets


def poison_strategy_for_multi_label_split(dataset, client_dataset_len_list, num_clients, label_list, random_ratio,
                                       **kwargs):
    all_subsets = []
    if len(label_list) < num_clients:
        print('Note: lesser labels for clients than number of requested clients, may consist of same label for '
              'multiple clients')

    for i in range(num_clients):
        # target_label = label_list[i]  # Choose the label you want to create a subset for
        limit_per_client = client_dataset_len_list[i]
        random_count = int(limit_per_client * random_ratio)

        if random_count > limit_per_client:
            print('random count is above 100%. Setting 100% randomness.')
            random_count = limit_per_client

        fixed_count = limit_per_client - random_count
        print('processing client: ', i)

        # subset_indices = [idx for idx, (_, label) in enumerate(dataset) if label == target_label][:fixed_count]
        # count = 0
        # for idx, (_, label) in enumerate(dataset):
        #     if label == target_label and count <= fixed_count:
        #         subset_indices.append(idx)
        #         count += 1
        #         #if count == fixed_count:
        #              #break
        if len(label_list) < num_clients:
            target_label = label_list[i % len(label_list)]  # since num_clients exceed the no. of labels,
            # multiple clients will get the same label
            print('assigning label with 1 for', target_label)
            all_indices = [idx for idx, (_, label) in enumerate(dataset) if label[i % len(label_list)] == 1]
            if fixed_count * num_clients > len(
                    all_indices):  # check if all possible labels are not sufficient for all shared clients
                fixed_count = len(all_indices) // num_clients  # distribute data among all clients if not sufficient
            subset_indices = all_indices[i * fixed_count:(i + 1) * fixed_count]

        else:
            # target_label = label_list[i]  # Choose the label you want to create a subset for
            all_indices = [idx for idx, (_, label) in enumerate(dataset) if label[i] == 1]
            if fixed_count * num_clients > len(
                    all_indices):  # check if all possible labels are not sufficient for all shared clients
                fixed_count = len(all_indices) // num_clients
            subset_indices = all_indices[:fixed_count]

        random_indices = random.sample(range(1, len(dataset)), random_count)

        subset_indices.extend(random_indices)
        random.shuffle(subset_indices)
        subset_dataset = Subset(dataset, subset_indices)
        target_clients = kwargs.get('target_clients')
        if target_clients is None:
            print('no poisoning for client: ', i)
            all_subsets.append(subset_dataset)

        elif i in target_clients:
            if kwargs.get('poison_type') == 'random_swap':
                print('perform random swap poisoning')
                subset_poisoned = random_swap(subset_dataset, kwargs.get('poison_ratio'))

            elif kwargs.get('poison_type') == 'random_poison':
                print('performing random poisoning of labels')
                subset_poisoned = random_poison(subset_dataset, kwargs.get('poison_ratio'), label_list)

            elif kwargs.get('poison_type') == 'target_poison':
                print('performing targeted poisoning of labels')
                subset_poisoned = target_poison(subset_dataset, kwargs.get('poison_ratio'), kwargs.get('target_label'))
            else:
                raise ValueError('Keyword argument for poison type is invalid.')

            all_subsets.append(subset_poisoned)

        else:
            print('no poisoning for client: ', i)
            all_subsets.append(subset_dataset)

    return all_subsets


