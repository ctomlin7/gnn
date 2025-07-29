"""Functions for dataloaders here. Take advantage of objects `paths.py` to load data from S3."""
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.loader import DataLoader

from data.preprocess import MolecularGraphDataset


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


def adversarial_split(data_list, test_size=0.2, random_state=42):
    # Step 1: Group data by adversarial_key
    key_to_data = defaultdict(list)
    for data in data_list:
        key_to_data[data.adversarial_key].append(data)

    # Step 2: Split keys into train and test
    all_keys = list(key_to_data.keys())
    random.Random(random_state).shuffle(all_keys)

    n_test = int(len(all_keys) * test_size)
    test_keys = set(all_keys[:n_test])
    train_keys = set(all_keys[n_test:])

    # Step 3: Split data based on key grouping
    train_list = []
    test_list = []

    for key in train_keys:
        train_list.extend(key_to_data[key])

    for key in test_keys:
        test_list.extend(key_to_data[key])

    assert train_keys.isdisjoint(test_keys), "Overlapping keys detected between train and test sets!"

    return train_list, test_list

def train_test_val_split(dataset):
    train, test_val = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
    test, val = train_test_split(test_val, test_size=0.5, random_state=42, shuffle=True)
    return train, test, val

def pe_transform(data_list, pe_dim):
    transform = AddLaplacianEigenvectorPE(k=pe_dim, attr_name='pe', is_undirected=True) 
    # choose k based on performance (higher=capture more complexity) true because bonds are undirected

    # Data List Transformation (appending pe)
    data_list_transformed = []
    for graphs in data_list:
        data_transformed = transform(graphs)
        data_list_transformed.append(data_transformed)
    
    return data_list_transformed

def load_data(csv_path, batch_size=32, y_class=None, shuffle=True, apply_transform=False, pe_dim=None, device=device):
    molecule_data = MolecularGraphDataset(csv_path, y_class)
    data_list = molecule_data.create_pytorch_geometric_graph_data_list_from_smiles_and_labels()
    
    if apply_transform == True:
        data_list = pe_transform(data_list, pe_dim)

    if y_class != None:
        train_list, test_list = adversarial_split(data_list, test_size=0.2, random_state=42)
    else:
        train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42, shuffle=True)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader

def get_property_labels(csv_path):
    molecules = MolecularGraphDataset(csv_path)
    property_labels = list(molecules.df.columns[2:])
    return property_labels