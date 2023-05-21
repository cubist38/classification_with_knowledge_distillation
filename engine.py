import torch

def train_one_epoch(model, 
                    dataloader, 
                    criterion,
                    optimizer, 
                    device):
    model.train()
    for samples, targets in dataloader:
        samples = samples.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * samples.size(0)
            total_correct += (outputs.argmax(dim = 1) == targets).sum().item()
            total_samples += samples.size(0)
    return total_loss / total_samples, total_correct / total_samples