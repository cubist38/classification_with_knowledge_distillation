import torch.nn as nn
import torch.nn.functional as F
import torch

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, criterion, temperature=1.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_loss = nn.KLDivLoss(reduction = 'batchmean')  
        self.criterion = criterion

    def forward(self, student_outputs, teacher_outputs, targets):
        losses = self.criterion(student_outputs, targets)
        soft_targets = F.softmax(teacher_outputs/ self.temperature, dim = 1)
        hard_targets = F.log_softmax(student_outputs / self.temperature, dim = 1)
        kd_loss = self.distillation_loss(hard_targets, soft_targets)
        loss = (1 - self.alpha) * losses.item() + self.alpha * self.temperature * self.temperature * kd_loss
        return loss