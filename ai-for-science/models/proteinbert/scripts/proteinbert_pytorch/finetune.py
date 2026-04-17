import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import ProteinBERTModel
from .inference import tokenize_seqs


class ProteinDataset(Dataset):
    def __init__(self, token_ids, annotations, labels, sample_weights=None):
        self.token_ids = torch.from_numpy(token_ids).long()
        self.annotations = torch.from_numpy(annotations).float()
        self.labels = torch.from_numpy(labels).float()
        if sample_weights is not None:
            self.sample_weights = torch.from_numpy(sample_weights).float()
        else:
            self.sample_weights = torch.ones(len(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.token_ids[idx], self.annotations[idx],
                self.labels[idx], self.sample_weights[idx])


class FinetuneModel(nn.Module):
    def __init__(self, pretrained_model, output_type='binary',
                 n_classes=1, use_global=True, dropout_rate=0.5):
        super().__init__()
        self.pretrained = pretrained_model
        self.use_global = use_global
        self.output_type = output_type

        if use_global:
            hidden_dim = pretrained_model.d_hidden_global
        else:
            hidden_dim = pretrained_model.d_hidden_seq

        self.dropout = nn.Dropout(dropout_rate)

        if output_type == 'categorical':
            self.head = nn.Linear(hidden_dim, n_classes)
        elif output_type == 'binary':
            self.head = nn.Linear(hidden_dim, 1)
        elif output_type == 'numeric':
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_seq, input_annotations):
        hidden_seq = self.pretrained.seq_embedding(input_seq)
        hidden_global = F.gelu(self.pretrained.global_input_dense(input_annotations))

        for block in self.pretrained.blocks:
            hidden_seq, hidden_global = block(hidden_seq, hidden_global)

        if self.use_global:
            features = hidden_global
        else:
            features = hidden_seq

        features = self.dropout(features)
        logits = self.head(features)

        if self.output_type == 'binary':
            return logits.squeeze(-1)
        elif self.output_type == 'categorical':
            return logits
        else:
            return logits.squeeze(-1)


def encode_dataset(seqs, labels, n_annotations, seq_len, output_type='binary'):
    token_ids = tokenize_seqs(list(seqs), seq_len)
    annotations = np.zeros((len(seqs), n_annotations), dtype=np.float32)
    labels_arr = np.array(list(labels), dtype=np.float32)
    return token_ids, annotations, labels_arr


def train_epoch(model, dataloader, optimizer, device, output_type='binary'):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        token_ids, annots, labels, weights = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(token_ids, annots)

        if output_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        elif output_type == 'categorical':
            loss = F.cross_entropy(logits, labels.long(), reduction='none')
        elif output_type == 'numeric':
            loss = F.mse_loss(logits, labels, reduction='none')

        loss = (loss * weights).sum() / weights.sum().clamp(min=1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * weights.sum().item()
        n_samples += weights.sum().item()

    return total_loss / max(n_samples, 1)


def evaluate(model, dataloader, device, output_type='binary'):
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            token_ids, annots, labels, weights = [x.to(device) for x in batch]
            logits = model(token_ids, annots)

            if output_type == 'binary':
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                preds = torch.sigmoid(logits)
            elif output_type == 'categorical':
                loss = F.cross_entropy(logits, labels.long(), reduction='none')
                preds = logits.argmax(dim=-1).float()
            elif output_type == 'numeric':
                loss = F.mse_loss(logits, labels, reduction='none')
                preds = logits

            loss = (loss * weights).sum() / weights.sum().clamp(min=1)
            total_loss += loss.item() * weights.sum().item()
            n_samples += weights.sum().item()

            mask = weights > 0
            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(labels[mask].cpu().numpy())

    avg_loss = total_loss / max(n_samples, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, all_preds, all_labels


def finetune_npu(pretrained_model, n_annotations, train_seqs, train_labels,
                 valid_seqs, valid_labels, device='npu:0',
                 output_type='binary', n_classes=2, use_global=True,
                 seq_len=512, batch_size=32, max_epochs=40,
                 lr=1e-4, frozen_lr=1e-2, dropout_rate=0.5,
                 begin_with_frozen=True, patience=2,
                 n_final_epochs=1, final_seq_len=1024, final_lr=1e-5):

    ft_model = FinetuneModel(pretrained_model, output_type=output_type,
                             n_classes=n_classes, use_global=use_global,
                             dropout_rate=dropout_rate).to(device)

    train_tokens, train_annots, train_y = encode_dataset(
        train_seqs, train_labels, n_annotations, seq_len, output_type)
    valid_tokens, valid_annots, valid_y = encode_dataset(
        valid_seqs, valid_labels, n_annotations, seq_len, output_type)

    train_ds = ProteinDataset(train_tokens, train_annots, train_y)
    valid_ds = ProteinDataset(valid_tokens, valid_annots, valid_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    def run_stage(epochs, learning_rate, freeze_pretrained=False, current_seq_len=None):
        if freeze_pretrained:
            for param in ft_model.pretrained.parameters():
                param.requires_grad = False
            ft_model.head.requires_grad_(True)
            ft_model.dropout.requires_grad_(True)
        else:
            for param in ft_model.parameters():
                param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           ft_model.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1, factor=0.25, min_lr=1e-5, verbose=True)

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(ft_model, train_dl, optimizer, device, output_type)
            val_loss, val_preds, val_labels = evaluate(ft_model, valid_dl, device, output_type)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            frozen_tag = ' [frozen]' if freeze_pretrained else ''
            print(f'  Epoch {epoch}/{epochs}{frozen_tag} - '
                  f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, '
                  f'lr: {current_lr:.2e}')

            if output_type == 'binary':
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(val_labels, val_preds)
                    print(f'           val_auc: {auc:.4f}')
                except ValueError:
                    pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f'  Early stopping at epoch {epoch}')
                    break

        if best_state is not None:
            ft_model.load_state_dict(best_state)

    if begin_with_frozen:
        print('\nStage 1: Training with frozen pretrained layers...')
        run_stage(max_epochs, frozen_lr, freeze_pretrained=True)

    print('\nStage 2: Training all layers...')
    run_stage(max_epochs, lr, freeze_pretrained=False)

    if n_final_epochs > 0:
        print(f'\nStage 3: Final training with seq_len={final_seq_len}...')
        final_batch = max(int(batch_size / (final_seq_len / seq_len)), 1)

        train_tokens_f, train_annots_f, train_y_f = encode_dataset(
            train_seqs, train_labels, n_annotations, final_seq_len, output_type)
        valid_tokens_f, valid_annots_f, valid_y_f = encode_dataset(
            valid_seqs, valid_labels, n_annotations, final_seq_len, output_type)

        nonlocal_hack = [train_dl, valid_dl]
        train_ds_f = ProteinDataset(train_tokens_f, train_annots_f, train_y_f)
        valid_ds_f = ProteinDataset(valid_tokens_f, valid_annots_f, valid_y_f)
        train_dl = DataLoader(train_ds_f, batch_size=final_batch, shuffle=True, num_workers=0)
        valid_dl = DataLoader(valid_ds_f, batch_size=final_batch, shuffle=False, num_workers=0)

        run_stage(n_final_epochs, final_lr, freeze_pretrained=False)

        train_dl, valid_dl = nonlocal_hack

    return ft_model
