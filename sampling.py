import numpy as np

def noniid_dirichlet_label(dataset, terminal_number, alpha):
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    label = np.array(label)
    
    access_point_number = len(np.unique(label))
    example_per_label = []
    for i in range(access_point_number):
        example_per_label.append(int(np.argwhere(label == i).shape[0]))

    multinomial_value = []
    for i in range(terminal_number):
        proportion = np.random.dirichlet(alpha * np.ones(access_point_number))
        multinomial_value.append(proportion)
        
    multinomial_value = np.array(multinomial_value)

    sample_index = []
    for k in range(access_point_number):
        label_index_set = np.where(label == k)[0]
        np.random.shuffle(label_index_set)
        sample_index.append(label_index_set)

    sample_index = np.array(sample_index, dtype=object)

    example_per_client = int(label.shape[0] / terminal_number)
    client_sample = [[] for _ in range(terminal_number)]
    count = np.zeros(access_point_number).astype(int)

    for k in range(terminal_number):
        for i in range(example_per_client):
            sample_label = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
            label_index = sample_index[sample_label]
            client_sample[k].append(label_index[count[sample_label]])
            count[sample_label] += 1
            if count[sample_label] == example_per_label[sample_label]:
                multinomial_value[:, sample_label] = 0
                if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                    multinomial_value = (
                        multinomial_value /
                        multinomial_value.sum(axis=1)[:, None])
    
    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(np.array(client_sample[i]))
        sample = np.array(client_sample[i])
        user_dictionary[i] = sample
    
    # first 9 sample's label of each terminal
    """
    for i in range(terminal_number):
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                print(f'{i + 1}: {label[sample]}')
    """
    
    return user_dictionary

def noniid_dirichlet_feature(dataset, application_id_set, terminal_number, alpha):
    feature = np.array(application_id_set)
    
    application_number = len(np.unique(feature))
    example_per_feature = []
    for i in range(application_number):
        example_per_feature.append(int(np.argwhere(feature == i).shape[0]))

    multinomial_value = []
    for i in range(terminal_number):
        proportion = np.random.dirichlet(alpha * np.ones(application_number))
        multinomial_value.append(proportion)
        
    multinomial_value = np.array(multinomial_value)

    sample_index = []
    for k in range(application_number):
        feature_index_set = np.where(feature == k)[0]
        np.random.shuffle(feature_index_set)
        sample_index.append(feature_index_set)

    sample_index = np.array(sample_index, dtype=object)

    example_per_client = int(feature.shape[0] / terminal_number)
    client_sample = [[] for _ in range(terminal_number)]
    count = np.zeros(application_number).astype(int)

    for k in range(terminal_number):
        for i in range(example_per_client):
            sample_feature = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
            feature_index = sample_index[sample_feature]
            client_sample[k].append(feature_index[count[sample_feature]])
            count[sample_feature] += 1
            if count[sample_feature] == example_per_feature[sample_feature]:
                multinomial_value[:, sample_feature] = 0
                if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                    multinomial_value = (
                        multinomial_value /
                        multinomial_value.sum(axis=1)[:, None])
    
    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(client_sample[i])
        sample = np.array(client_sample[i])
        user_dictionary[i] = sample
    
    # first 9 sample's feature and label of each terminal
    '''
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    label = np.array(label)

    for i in range(5):
        print('\n')
        print(f'-----{i + 1}-----')
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                # print(feature[sample], label[sample])
                print(feature[sample])
    '''
    
    return user_dictionary

def noniid_dirichlet_quantity(dataset, min_required_size, max_required_size, terminal_number, alpha):
    idxs = np.random.permutation(len(dataset))
    
    proportions = np.random.dirichlet(np.repeat(alpha, terminal_number))
    proportions = proportions / proportions.sum()
    proportions = [(min_required_size + 1) / len(idxs) if i < (min_required_size + 1) / len(idxs) else i for i
                in proportions]
    proportions = [(max_required_size + 1) / len(idxs) if i > (max_required_size + 1) / len(idxs) else i for i
                in proportions]
    proportions = [i / sum(proportions) for i in proportions]

    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    idx_batch = np.split(idxs, proportions)

    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(idx_batch[i])
        sample = np.array(idx_batch[i])
        user_dictionary[i] = sample

    return user_dictionary

def noniid_dirichlet_label_feature(dataset, application_id_set, terminal_number, alpha):
    # feature
    feature = np.array(application_id_set)
    
    application_number = len(np.unique(feature))
    example_per_feature = []
    for i in range(application_number):
        example_per_feature.append(int(np.argwhere(feature == i).shape[0]))

    multinomial_value = []
    for i in range(application_number):
        proportion = np.random.dirichlet(alpha * np.ones(application_number))
        multinomial_value.append(proportion)
        
    multinomial_value = np.array(multinomial_value)

    sample_index = []
    for k in range(application_number):
        feature_index_set = np.where(feature == k)[0]
        np.random.shuffle(feature_index_set)
        sample_index.append(feature_index_set)

    sample_index = np.array(sample_index, dtype=object)

    example_per_client = int(feature.shape[0] / application_number)
    client_sample_feature = [[] for _ in range(application_number)]
    count = np.zeros(application_number).astype(int)

    for k in range(application_number):
        for i in range(example_per_client):
            sample_feature = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
            feature_index = sample_index[sample_feature]
            client_sample_feature[k].append(feature_index[count[sample_feature]])
            count[sample_feature] += 1
            if count[sample_feature] == example_per_feature[sample_feature]:
                multinomial_value[:, sample_feature] = 0
                if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                    multinomial_value = (
                        multinomial_value /
                        multinomial_value.sum(axis=1)[:, None])
    
    '''
    # first 9 sample's feature and label of each terminal

    for i in range(application_number):
        print('\n')
        print(f'-----{i + 1}-----')
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                print(feature[sample])
    '''
    
    # label
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    
    label_set = [[] for _ in range(application_number)]
    for i in range(application_number):
        for idx in client_sample_feature[i]:
            label_set[i].append(label[idx])
    
    access_point_number = len(np.unique(label))
    each_terminal_number = int(terminal_number / application_number)
    client_sample_label_set = [[] for _ in range(application_number)]
    for j, label in enumerate(label_set):
        label = np.array(label)
        example_per_label = []
        for i in range(access_point_number):
            example_per_label.append(int(np.argwhere(label == i).shape[0]))

        multinomial_value = []
        for i in range(each_terminal_number):
            proportion = np.random.dirichlet(alpha * np.ones(access_point_number))
            multinomial_value.append(proportion)
            
        multinomial_value = np.array(multinomial_value)

        for i in range(access_point_number):
            if example_per_label[i] == 0:
                multinomial_value[:, i] = 0
                multinomial_value = (
                    multinomial_value /
                    multinomial_value.sum(axis=1)[:, None])

        sample_index = []
        for k in range(access_point_number):
            label_index_set = np.where(label == k)[0]
            np.random.shuffle(label_index_set)
            sample_index.append(label_index_set)

        sample_index = np.array(sample_index, dtype=object)

        example_per_client = int(label.shape[0] / each_terminal_number)
        count = np.zeros(access_point_number).astype(int)

        for k in range(each_terminal_number):
            client_sample_label = []
            for i in range(example_per_client):
                sample_label = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
                label_index = sample_index[sample_label]
                client_sample_label.append(label_index[count[sample_label]])
                count[sample_label] += 1
                if count[sample_label] == example_per_label[sample_label]:
                    multinomial_value[:, sample_label] = 0
                    if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                        multinomial_value = (
                            multinomial_value /
                            multinomial_value.sum(axis=1)[:, None])
        
            client_sample_label_set[j].append(client_sample_label)
    
    client_sample = [[] for _ in range(terminal_number)]
    i = 0
    for j, sample_set in enumerate(client_sample_label_set):
        for idx_set in sample_set:
            for idx in idx_set:
                client_sample[i].append(client_sample_feature[j][idx])
            i += 1

    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(np.array(client_sample[i]))
        sample = np.array(client_sample[i])
        user_dictionary[i] = sample
    
    # first 9 sample's feature and label of each terminal
    '''
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    label = np.array(label)

    for i in range(100):
        print('\n')
        print(f'-----{i + 1}-----')
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                # print(feature[sample], label[sample])
                print(feature[sample], label[sample])
    '''

    return user_dictionary

def noniid_dirichlet_label_quantity(dataset, min_required_size, max_required_size, terminal_number, alpha):
    # quantity
    idxs = np.random.permutation(len(dataset))
    
    proportions = np.random.dirichlet(np.repeat(alpha, terminal_number))
    proportions = proportions / proportions.sum()
    proportions = [(min_required_size + 1) / len(idxs) if i < (min_required_size + 1) / len(idxs) else i for i
                in proportions]
    proportions = [(max_required_size + 1) / len(idxs) if i > (max_required_size + 1) / len(idxs) else i for i
                in proportions]
    proportions = [i / sum(proportions) for i in proportions]

    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    idx_batch = np.split(idxs, proportions)
    data_number = [len(idx_batch[i]) for i in range(len(idx_batch))]

    # label
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    label = np.array(label)
    
    access_point_number = len(np.unique(label))
    example_per_label = []
    for i in range(access_point_number):
        example_per_label.append(int(np.argwhere(label == i).shape[0]))

    multinomial_value = []
    for i in range(terminal_number):
        proportion = np.random.dirichlet(alpha * np.ones(access_point_number))
        multinomial_value.append(proportion)
        
    multinomial_value = np.array(multinomial_value)

    sample_index = []
    for k in range(access_point_number):
        label_index_set = np.where(label == k)[0]
        np.random.shuffle(label_index_set)
        sample_index.append(label_index_set)

    sample_index = np.array(sample_index, dtype=object)

    client_sample = [[] for _ in range(terminal_number)]
    count = np.zeros(access_point_number).astype(int)

    for k in range(terminal_number):
        for i in range(data_number[k]):
            sample_label = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
            label_index = sample_index[sample_label]
            client_sample[k].append(label_index[count[sample_label]])
            count[sample_label] += 1
            if count[sample_label] == example_per_label[sample_label]:
                multinomial_value[:, sample_label] = 0
                if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                    multinomial_value = (
                        multinomial_value /
                        multinomial_value.sum(axis=1)[:, None])
    
    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(np.array(client_sample[i]))
        sample = np.array(client_sample[i])
        user_dictionary[i] = sample
    
    # first 9 sample's label of each terminal
    """
    for i in range(terminal_number):
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                print(f'{i + 1}: {label[sample]}')
    """
    
    return user_dictionary

def noniid_dirichlet_feature_quantity(dataset, application_id_set, min_required_size, max_required_size, terminal_number, alpha):
    # quantity
    idxs = np.random.permutation(len(dataset))
    
    proportions = np.random.dirichlet(np.repeat(alpha, terminal_number))
    proportions = proportions / proportions.sum()
    proportions = [(min_required_size + 1) / len(idxs) if i < (min_required_size + 1) / len(idxs) else i for i
                in proportions]
    proportions = [(max_required_size + 1) / len(idxs) if i > (max_required_size + 1) / len(idxs) else i for i
                in proportions]
    proportions = [i / sum(proportions) for i in proportions]

    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    idx_batch = np.split(idxs, proportions)
    data_number = [len(idx_batch[i]) for i in range(len(idx_batch))]

    # feature
    feature = np.array(application_id_set)
    
    application_number = len(np.unique(feature))
    example_per_feature = []
    for i in range(application_number):
        example_per_feature.append(int(np.argwhere(feature == i).shape[0]))

    multinomial_value = []
    for i in range(terminal_number):
        proportion = np.random.dirichlet(alpha * np.ones(application_number))
        multinomial_value.append(proportion)
        
    multinomial_value = np.array(multinomial_value)

    sample_index = []
    for k in range(application_number):
        feature_index_set = np.where(feature == k)[0]
        np.random.shuffle(feature_index_set)
        sample_index.append(feature_index_set)

    sample_index = np.array(sample_index, dtype=object)

    client_sample = [[] for _ in range(terminal_number)]
    count = np.zeros(application_number).astype(int)

    for k in range(terminal_number):
        for i in range(data_number[k]):
            sample_feature = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
            feature_index = sample_index[sample_feature]
            client_sample[k].append(feature_index[count[sample_feature]])
            count[sample_feature] += 1
            if count[sample_feature] == example_per_feature[sample_feature]:
                multinomial_value[:, sample_feature] = 0
                if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                    multinomial_value = (
                        multinomial_value /
                        multinomial_value.sum(axis=1)[:, None])
    
    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(np.array(client_sample[i]))
        sample = np.array(client_sample[i])
        user_dictionary[i] = sample
    
    '''
    # first 9 sample's feature and label of each terminal
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    label = np.array(label)

    for i in range(100):
        print('\n')
        print(f'-----{i + 1}-----')
        print(len(user_dictionary[i]))
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                print(feature[sample], label[sample])
                # print(feature[sample])
    '''
    
    return user_dictionary

def noniid_dirichlet_label_feature_quantity(dataset, application_id_set, min_required_size, max_required_size, terminal_number, alpha):
    # feature
    feature = np.array(application_id_set)
    
    application_number = len(np.unique(feature))
    example_per_feature = []
    for i in range(application_number):
        example_per_feature.append(int(np.argwhere(feature == i).shape[0]))

    multinomial_value = []
    for i in range(application_number):
        proportion = np.random.dirichlet(alpha * np.ones(application_number))
        multinomial_value.append(proportion)
        
    multinomial_value = np.array(multinomial_value)

    sample_index = []
    for k in range(application_number):
        feature_index_set = np.where(feature == k)[0]
        np.random.shuffle(feature_index_set)
        sample_index.append(feature_index_set)

    sample_index = np.array(sample_index, dtype=object)

    example_per_client = int(feature.shape[0] / application_number)
    client_sample_feature = [[] for _ in range(application_number)]
    count = np.zeros(application_number).astype(int)

    for k in range(application_number):
        for i in range(example_per_client):
            sample_feature = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
            feature_index = sample_index[sample_feature]
            client_sample_feature[k].append(feature_index[count[sample_feature]])
            count[sample_feature] += 1
            if count[sample_feature] == example_per_feature[sample_feature]:
                multinomial_value[:, sample_feature] = 0
                if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                    multinomial_value = (
                        multinomial_value /
                        multinomial_value.sum(axis=1)[:, None])
    
    '''
    # first 9 sample's feature and label of each terminal

    for i in range(application_number):
        print('\n')
        print(f'-----{i + 1}-----')
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                print(feature[sample])
    '''
    
    # label
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    
    label_set = [[] for _ in range(application_number)]
    for i in range(application_number):
        for idx in client_sample_feature[i]:
            label_set[i].append(label[idx])
    
    access_point_number = len(np.unique(label))
    each_terminal_number = int(terminal_number / application_number)
    client_sample_label_set = [[] for _ in range(application_number)]
    for j, label in enumerate(label_set):
        # quantity
        idxs = np.random.permutation(int(len(dataset) / application_number))
        
        proportions = np.random.dirichlet(np.repeat(alpha, each_terminal_number))
        proportions = proportions / proportions.sum()
        proportions = [(min_required_size + 1) / len(idxs) if i < (min_required_size + 1) / len(idxs) else i for i
                    in proportions]
        proportions = [(max_required_size + 1) / len(idxs) if i > (max_required_size + 1) / len(idxs) else i for i
                    in proportions]
        proportions = [i / sum(proportions) for i in proportions]

        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        idx_batch = np.split(idxs, proportions)
        data_number = [len(idx_batch[i]) for i in range(len(idx_batch))]

        # label continue...
        label = np.array(label)
        example_per_label = []
        for i in range(access_point_number):
            example_per_label.append(int(np.argwhere(label == i).shape[0]))

        multinomial_value = []
        for i in range(each_terminal_number):
            proportion = np.random.dirichlet(alpha * np.ones(access_point_number))
            multinomial_value.append(proportion)
            
        multinomial_value = np.array(multinomial_value)

        for i in range(access_point_number):
            if example_per_label[i] == 0:
                multinomial_value[:, i] = 0
                multinomial_value = (
                    multinomial_value /
                    multinomial_value.sum(axis=1)[:, None])

        sample_index = []
        for k in range(access_point_number):
            label_index_set = np.where(label == k)[0]
            np.random.shuffle(label_index_set)
            sample_index.append(label_index_set)

        sample_index = np.array(sample_index, dtype=object)

        count = np.zeros(access_point_number).astype(int)

        for k in range(each_terminal_number):
            client_sample_label = []
            for i in range(data_number[k]):
                sample_label = np.argwhere(np.random.multinomial(1, multinomial_value[k, :]) == 1)[0][0]
                label_index = sample_index[sample_label]
                client_sample_label.append(label_index[count[sample_label]])
                count[sample_label] += 1
                if count[sample_label] == example_per_label[sample_label]:
                    multinomial_value[:, sample_label] = 0
                    if sum(multinomial_value.sum(axis=1)[:, None][0]) != 0:
                        multinomial_value = (
                            multinomial_value /
                            multinomial_value.sum(axis=1)[:, None])
        
            client_sample_label_set[j].append(client_sample_label)
    
    client_sample = [[] for _ in range(terminal_number)]
    i = 0
    for j, sample_set in enumerate(client_sample_label_set):
        for idx_set in sample_set:
            for idx in idx_set:
                client_sample[i].append(client_sample_feature[j][idx])
            i += 1

    user_dictionary = {}
    for i in range(terminal_number):
        np.random.shuffle(np.array(client_sample[i]))
        sample = np.array(client_sample[i])
        user_dictionary[i] = sample
    
    '''
    # first 9 sample's feature and label of each terminal
    label = []
    for i in range(len(dataset)):
        label.append(dataset[i][1])
    label = np.array(label)

    for i in range(100):
        print('\n')
        print(f'-----{i + 1}-----')
        print(len(user_dictionary[i]))
        for j, sample in enumerate(user_dictionary[i]):
            if 0 <= j and j < 9:
                # print(feature[sample], label[sample])
                print(feature[sample], label[sample])
    '''

    return user_dictionary