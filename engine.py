import torch
import tqdm

def train_one_epoch(model, 
                    dataloader, 
                    criterion,
                    optimizer, 
                    device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for samples, targets in tqdm.tqdm(dataloader, total = len(dataloader)):
        samples = samples.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_samples += samples.size(0)
    return running_loss/total_samples

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for samples, targets in tqdm.tqdm(dataloader, total = len(dataloader)):
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim = 1) == targets).sum().item()
            total_samples += samples.size(0)
    return total_loss / total_samples, total_correct / total_samples