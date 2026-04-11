import torch
import torch.nn as nn

class EWC:
    def __init__(self, model, dataloader, device='cpu', fisher_n=100):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.fisher_n = fisher_n
        # Save a copy of model parameters for penalty
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self.compute_fisher()

    def compute_fisher(self):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        for i, batch in enumerate(self.dataloader):
            if i >= self.fisher_n:
                break

            self.model.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device).long()

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Compute loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2

        # Average over number of batches used
        for n in fisher:
            fisher[n] /= min(self.fisher_n, len(self.dataloader))

        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss
