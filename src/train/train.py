import numpy as np
from sklearn.metrics import r2_score, accuracy_score
import torch


def train(model, loader, optimizer, device):
    model.to(device)
    model.train()
    total_loss = 0
    losses = []
    outputs = []
    targets = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)
        y_true = model.prepare_targets(batch, device)
        loss = model.compute_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Collect data for analysis
        losses.append(loss.item())
        outputs.append(y_pred.detach().cpu().numpy())
        targets.append(y_true.detach().cpu().numpy())

    total_loss /= len(loader)
    
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    if model.task == 'regression':
        train_scores = []
        if targets.ndim == 1:
            targets = targets[:, None]
            outputs = outputs[:, None]

        for i in range(targets.shape[1]):
            target_i = targets[:, i]
            output_i = outputs[:, i]
            train_scores.append(r2_score(target_i, output_i))

        avg_acc = r2_score(targets.flatten(), outputs.flatten())
    elif model.task == 'classification':
        pred_classes = np.argmax(outputs, axis=1)
        avg_acc = accuracy_score(targets.flatten(), pred_classes)

        num_classes = outputs.shape[1]
        train_scores = []
        for i in range(num_classes):
            class_indices = targets.flatten() == i
            if np.any(class_indices):
                acc = accuracy_score(targets[class_indices], pred_classes[class_indices])
                train_scores.append(acc)
            else:
                train_scores.append(None)
    else:
        raise ValueError('Model task is not recognized.')
         
    return total_loss, avg_acc, train_scores

@torch.no_grad()
def test(model, loader, device):
    model.to(device)
    model.eval()
    total_loss = 0
    losses = []
    outputs = []
    targets = []

    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch)
        y_true = model.prepare_targets(batch, device)
        loss = model.compute_loss(y_pred, y_true)     
        total_loss += loss.item()
        
        # Collect data for analysis
        losses.append(loss.item())
        outputs.append(y_pred.detach().cpu().numpy())
        targets.append(y_true.detach().cpu().numpy())

    total_loss /= len(loader)

    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    if model.task == 'regression':
        test_scores = []
        if targets.ndim == 1:
            targets = targets[:, None]
            outputs = outputs[:, None]

        for i in range(targets.shape[1]):
            target_i = targets[:, i]
            output_i = outputs[:, i]
            test_scores.append(r2_score(target_i, output_i))

        avg_acc = r2_score(targets.flatten(), outputs.flatten())
    elif model.task == 'classification':
        pred_classes = np.argmax(outputs, axis=1)
        avg_acc = accuracy_score(targets.flatten(), pred_classes)

        num_classes = outputs.shape[1]
        test_scores = []
        for i in range(num_classes):
            class_indices = targets.flatten() == i
            if np.any(class_indices):
                acc = accuracy_score(targets[class_indices], pred_classes[class_indices])
                test_scores.append(acc)
            else:
                test_scores.append(None)
    else:
        raise ValueError('Model task is not recognized.')

    return total_loss, avg_acc, test_scores