import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from data.loaders import load_data
from models.models import GPS
from train.train import train, test

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


def objective(trial):
    num_layers = trial.suggest_categorical("Num Layers", [2, 4, 6])
    channels = trial.suggest_categorical("Hidden Channels", [16, 32, 64])
    mpnn_type = trial.suggest_categorical("Message Passing Type", ["GAT", "GINE", "GatedGCN"])
    num_heads = trial.suggest_categorical("Num Heads", [4, 8])
    # pe_dim = trial.suggest_categorical("PE Dim", [20, 35, 50])
    
    out_channels = 100
    attn_type = "multihead"

    gps_model = GPS(channels, 
                    out_channels, 
                    node_feats, 
                    edge_feats, 
                    mpnn_type, 
                    num_heads, 
                    num_layers, 
                    attn_type, 
                    pe_dim)

    gps_model = gps_model.to(device)
    optimizer = optim.Adam(gps_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
    criterion = nn.MSELoss()
    num_epochs = 100

    # Model Run
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_acc, _ = train(gps_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = test(gps_model, val_loader, criterion, device)
        scheduler.step(val_loss)

        losses.append([epoch+1, train_loss, val_loss])
        accuracies.append([epoch+1, train_acc, val_acc])

    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    args = parser.parse_args()

    filepath = args.filepath
    batch_size = 32
    pe_dim = 20

    print("Loading Data...")
    train_loader, test_loader, val_loader = load_data(filepath, 
                                                       batch_size=batch_size, 
                                                       shuffle=True, 
                                                       apply_transform=True, 
                                                       pe_dim=pe_dim,
                                                       device=device)
    print("Data has finished loading")

    for batch in train_loader:
        node_feats = batch.x.shape[1]
        edge_feats = batch.edge_attr.shape[1]
        break

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="model:GAT-testing:num_layers/channels/mpnn_layer/num_heads-ouput:test_loss",
        direction="minimize"
    )

    study.optimize(objective, n_trials=50)