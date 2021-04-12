"""Utils for train and test"""
import torch
import numpy as np


def test(test_loader, model, criterion, device):
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        _, pred_label = torch.max(output.data, 1)
        total_cnt = data.data.size()[0]
        correct_cnt = (pred_label == target.data).sum()
        test_acc = correct_cnt * 1.0 / total_cnt

    return test_acc


def train_one_epoch(epoch, n_epochs, train_loader, test_loader, model,
                    optimizer, criterion, disp_freq, device):
    batch_loss, batch_acc = [], []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())

        if (batch_idx + 1) % disp_freq == 0 or (batch_idx +
                                                1) == len(train_loader):
            test_acc = test(test_loader, model, criterion, device)
            batch_acc.append(test_acc)
            print(
                "Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}"
                .format(epoch + 1, n_epochs, batch_idx + 1, len(train_loader),
                        np.mean(batch_loss), np.mean(batch_acc)))

    return np.mean(batch_loss), np.mean(batch_acc)
