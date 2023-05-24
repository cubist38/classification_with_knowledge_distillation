import torch
import tqdm
from typing import Iterable

def train_one_epoch_kd(student_model: torch.nn.Module, 
                        teacher_model:  torch.nn.Module,
                        criterion: torch.nn.Module, data_loader: Iterable, 
                        optimizer: torch.optim.Optimizer, device: torch.device):
    student_model.train()
    running_loss = 0.0
    total_samples = 0
    for samples, targets in tqdm.tqdm(data_loader, total = len(data_loader)):
        samples = samples.to(device)
        targets = targets.to(device)
        student_outputs = student_model(samples)
        teacher_outputs = teacher_model(samples)
        loss = criterion(student_outputs, teacher_outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_samples += samples.size(0)
    return running_loss / total_samples

def eval_kd(student_model: torch.nn.Module,
            criterion: torch.nn.Module, 
            data_loader: Iterable,
            device: torch.device):
    student_model.eval()
    total_loss = 0.0
    total_samples = 0
    for samples, labels in tqdm.tqdm(data_loader, total = len(data_loader)):
        samples = samples.to(device)
        labels = labels.to(device)
        outputs = student_model(samples)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_samples += samples.size(0)
    return total_loss/total_samples

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
    total_samples = 0
    with torch.no_grad():
        for samples, targets in tqdm.tqdm(dataloader, total = len(dataloader)):
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_samples += samples.size(0)
    return total_loss / total_samples