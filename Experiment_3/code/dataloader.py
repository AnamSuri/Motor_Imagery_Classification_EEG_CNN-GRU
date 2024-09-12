


def load_and_combine_data(subjects):
    all_data = []
    all_labels = []
    for subject_id in subjects:
        for session in range(1, 6):  # Loop through all 5 sessions
            mat_data = sio.loadmat(f'path to mat file')
            data = mat_data['data']
            labels = mat_data['labels'].reshape(-1)
            data = data / np.max(data, axis=2, keepdims=True)  # Normalize data
            all_data.append(data)
            all_labels.append(labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels)
    return torch.tensor(all_data).float(), torch.tensor(all_labels - 1, dtype=torch.long)
