import argparse
import torch
import torch.optim as optim

from analysis.analysis import plot_results, plot_r2_scores, box_plot, box_plot_rmse, barplot_class_accuracy, epoch_plot 
from data.loaders import load_data, get_property_labels
from models.models import GPS, save_state
from models.warm_start import warm_start_classification_model
from train.train import train, test

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


def model_run(model, train_loader, test_loader, num_epochs, optimizer, scheduler, device):
    # Model Run
    losses = []
    accuracies = []
    scores = []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_scores = train(model, train_loader, optimizer, device)
        test_loss, test_acc, test_scores = test(model, test_loader, device)
        scheduler.step(test_loss)

        losses.append([epoch+1, train_loss, test_loss])
        accuracies.append([epoch+1, train_acc, test_acc])
        scores.append([epoch+1] + train_scores + test_scores)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.5f} \n"
            f"         Test Loss: {test_loss:.4f},  Test Accuracy: {test_acc:.5f}")
    
    return losses, accuracies, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('class_data')
    args = parser.parse_args()

    filepath = args.filepath
    batch_size = 32
    pe_dim = 20

    print("Loading Data...")
    train_loader, test_loader = load_data(filepath, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          apply_transform=True, 
                                          pe_dim=pe_dim,
                                          device=device)
    print("Data has finished loading")
    
    for batch in train_loader:
        num_node_feats = batch.x.shape[1]
        num_edge_feats = batch.edge_attr.shape[1]
        break

    # Parameter Initialization
    channels = 64
    out_channels_reg = 100
    mpnn_type = "GatedGCN"
    num_heads = 4
    num_layers = 4
    attn_type = "multihead"

    # Regression Model Initialization
    gps_reg_model = GPS(channels, out_channels_reg, num_node_feats, num_edge_feats, mpnn_type, num_heads, num_layers, attn_type, pe_dim, task='regression')
    gps_reg_model = gps_reg_model.to(device)
    optimizer = optim.Adam(gps_reg_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
    num_epochs_reg = 100

    # Model Run for Regression
    print("Running Regression Model")
    reg_losses, reg_accuracies, r2_scores = model_run(gps_reg_model, train_loader, test_loader, num_epochs_reg, optimizer, scheduler, device)

    # Plotting
    epoch_plot(reg_losses, reg_accuracies, save='../plots/gat/reg_epoch_plot.png')
    plot_results(reg_losses, save="../plots/gat/loss_plot_full.png")
    box_plot(r2_scores, get_property_labels(filepath), save="../plots/gat/boxplot_accs_full.png")
    box_plot_rmse(reg_losses, save="../plots/gat/boxplot_rmse_full.png")

    save_state(gps_reg_model, "gps_reg")
    print("Regression Model Saved!")


    # Load Classification Data
    class_data_filepath = args.class_data

    print("Loading Classification Data...")
    train_loader, test_loader = load_data(class_data_filepath,
                                          y_class=True, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          apply_transform=True, 
                                          pe_dim=pe_dim,
                                          device=device)
    print("Classification Data has finished loading")

    # Classification Output Channels
    out_channels_class = 2
    
    # Classification Model Initialization
    gps_class_model = GPS(channels, out_channels_class, num_node_feats, num_edge_feats, mpnn_type, num_heads, num_layers, attn_type, pe_dim, task='classification')
    gps_class_model = gps_class_model.to(device)
    # Warm-Start Model
    gps_class_model = warm_start_classification_model('gps_reg_state.pth', gps_class_model)

    optimizer = optim.Adam(gps_class_model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
    num_epochs_class = 10

    # Model Run for Classification
    print("Running Classification Model")
    class_losses, class_accuracies, class_scores = model_run(gps_class_model, train_loader, test_loader, num_epochs_class, optimizer, scheduler, device)

    # Plotting
    epoch_plot(class_losses, class_accuracies, save='../plots/gat/class_epoch_plot.png')
    barplot_class_accuracy(class_accuracies, class_labels=[0, 1, 2], save='../plots/gat/barplot_class_accs.png')

    save_state(gps_class_model, "gps_class")
    print("Classification Model Saved!")
    