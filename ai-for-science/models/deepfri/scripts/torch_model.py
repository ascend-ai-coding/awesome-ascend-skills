import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_layers import (
    MultiGraphConv, GraphConv, SAGEConv, NoGraphConv, ChebConv,
    SumPooling, FuncPredictor,
)

GCONV_LAYERS = {
    'MultiGraphConv': MultiGraphConv,
    'GraphConv': GraphConv,
    'SAGEConv': SAGEConv,
    'NoGraphConv': NoGraphConv,
    'ChebConv': ChebConv,
}


class LSTMLanguageModel(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=512):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        return torch.cat([out1, out2], dim=-1)


class DeepFRIGCN(nn.Module):
    def __init__(self, output_dim, n_channels=26, gc_dims=None, fc_dims=None,
                 drop=0.3, gc_layer='MultiGraphConv', lm_model=None):
        super().__init__()
        if gc_dims is None:
            gc_dims = [512, 512, 512]
        if fc_dims is None:
            fc_dims = [1024]

        self.lm_dim = 1024
        self.gc_layer_name = gc_layer
        GConv = GCONV_LAYERS.get(gc_layer, NoGraphConv)

        self.aa_embedding = nn.Linear(n_channels, self.lm_dim, bias=False)
        self.has_lm = lm_model is not None
        if self.has_lm:
            self.lm_model = lm_model
            for p in self.lm_model.parameters():
                p.requires_grad = False
            self.lm_embedding = nn.Linear(self.lm_dim, self.lm_dim, bias=True)

        self.gc_layers = nn.ModuleList()
        in_dim = self.lm_dim
        for dim in gc_dims:
            self.gc_layers.append(GConv(in_dim, dim, use_bias=False, activation='elu'))
            in_dim = dim

        self.sum_pool = SumPooling(axis=1)
        concat_dim = sum(gc_dims)

        self.fc_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_fc = concat_dim
        for i, dim in enumerate(fc_dims):
            self.fc_layers.append(nn.Linear(in_fc, dim))
            self.dropouts.append(nn.Dropout((i + 1) * drop))
            in_fc = dim

        self.func_predictor = FuncPredictor(in_fc, output_dim)

    def forward(self, cmap, seq):
        x_aa = self.aa_embedding(seq)
        if self.has_lm:
            x_lm = self.lm_embedding(self.lm_model(seq))
            x_aa = x_lm + x_aa
        x = F.relu(x_aa)

        gcnn_concat = []
        for gc in self.gc_layers:
            x = gc(x, cmap)
            gcnn_concat.append(x)

        if len(gcnn_concat) > 1:
            x = torch.cat(gcnn_concat, dim=-1)
        else:
            x = gcnn_concat[-1]

        x = self.sum_pool(x)

        for fc, drop in zip(self.fc_layers, self.dropouts):
            x = F.relu(fc(x))
            x = drop(x)

        return self.func_predictor(x)


class DeepCNN(nn.Module):
    def __init__(self, output_dim, n_channels=26, num_filters=None, filter_lens=None,
                 drop=0.3):
        super().__init__()
        if num_filters is None:
            num_filters = [256] * 16
        if filter_lens is None:
            filter_lens = [8, 16, 25, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]

        self.conv_layers = nn.ModuleList()
        for nf, fl in zip(num_filters, filter_lens):
            self.conv_layers.append(nn.Conv1d(n_channels, nf, fl, padding='same'))

        total_filters = sum(num_filters)
        self.bn = nn.BatchNorm1d(total_filters, eps=1e-3)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(2 * drop)
        self.func_predictor = FuncPredictor(total_filters, output_dim)

    def forward(self, seq):
        x = seq.transpose(1, 2)
        conv_outs = [conv(x) for conv in self.conv_layers]
        x = torch.cat(conv_outs, dim=1)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = x.max(dim=2)[0]
        x = self.drop2(x)
        return self.func_predictor(x)
