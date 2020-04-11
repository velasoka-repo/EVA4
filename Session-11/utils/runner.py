from tqdm import tqdm
import torch


def train(model, data_loader, optimizer, criterion, device, policy):
    model.train()
    correct_prediction = 0
    total_dataset = 0
    progress_bar = tqdm(iterable=data_loader, total=len(data_loader), position=0)
    for batch_id, (data, label) in enumerate(progress_bar):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        result = model(data)
        loss = criterion(result, label)

        loss.backward()
        optimizer.step()

        predicted = result.argmax(dim=1, keepdim=True)
        correct_prediction += (label == predicted.view_as(label)).sum().item()
        total_dataset += len(data)

        policy.step()

        progress_bar.set_description(
            f"Batch: {batch_id}, loss: {loss.item():.2f}, Train Accuracy: {100 * correct_prediction / total_dataset:.2f}")

    return 100 * correct_prediction / total_dataset


def validate(model, data_loader, device):
    model.eval()
    correct_prediction = 0
    total_dataset = 0
    progress_bar = tqdm(iterable=data_loader, total=len(data_loader), position=0)
    with torch.no_grad():
        for batch_id, (data, label) in enumerate(progress_bar):
            data, label = data.to(device), label.to(device)
            result = model(data)
            predicted = result.argmax(dim=1, keepdim=True)
            correct_prediction += (label == predicted.view_as(label)).sum().item()
            total_dataset += len(data)

            progress_bar.set_description(
                f"Batch: {batch_id}, Test Accuracy: {100 * correct_prediction / total_dataset:.2f}")
