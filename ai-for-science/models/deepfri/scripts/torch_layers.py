import torch
import torch.nn as nn
import torch.nn.functional as F


class SumPooling(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.sum(x, dim=self.axis)


class FuncPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.dense = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), self.output_dim, 2)
        return F.softmax(x, dim=-1)


class MultiGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, activation='elu'):
        super().__init__()
        self.output_dim = output_dim
        self.kernel = nn.Parameter(torch.empty(3 * input_dim, output_dim))
        nn.init.xavier_uniform_(self.kernel)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = None

    def _normalize(self, A, eps=1e-6):
        A = A.clone()
        n = A.size(-1)
        eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        A = A - torch.diag_embed(diag)
        A_hat = A + eye
        deg = A_hat.sum(dim=2)
        D_asymm = torch.diag_embed(1.0 / (eps + deg))
        D_symm = torch.diag_embed(1.0 / (eps + deg.sqrt()))
        return [A, torch.bmm(D_asymm, A_hat), torch.bmm(torch.bmm(D_symm, A_hat), D_symm)]

    def forward(self, features, adj):
        norms = self._normalize(adj)
        outputs = [torch.bmm(n, features) for n in norms]
        output = torch.cat(outputs, dim=-1)
        output = torch.matmul(output, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, activation='elu'):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.kernel)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = None

    def _normalize(self, A, eps=1e-6):
        A = A.clone()
        n = A.size(-1)
        eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        A = A - torch.diag_embed(diag)
        A_hat = A + eye
        D_hat = torch.diag_embed(1.0 / (eps + A_hat.sum(dim=2).sqrt()))
        return torch.bmm(torch.bmm(D_hat, A_hat), D_hat)

    def forward(self, features, adj):
        norm = self._normalize(adj)
        output = torch.bmm(norm, features)
        output = torch.matmul(output, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, activation='elu'):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(2 * input_dim, output_dim))
        nn.init.xavier_uniform_(self.kernel)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = None

    def _normalize(self, A, eps=1e-6):
        A = A.clone()
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        A = A - torch.diag_embed(diag)
        D = torch.diag_embed(1.0 / (eps + A.sum(dim=2)))
        return torch.bmm(D, A)

    def forward(self, features, adj):
        norm = self._normalize(adj)
        output = torch.bmm(norm, features)
        output = torch.cat([output, features], dim=-1)
        output = torch.matmul(output, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class NoGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, activation='elu'):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.kernel)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = None

    def forward(self, features, adj):
        n = adj.size(-1)
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        output = torch.bmm(eye.expand(features.size(0), -1, -1), features)
        output = torch.matmul(output, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class ChebConv(nn.Module):
    def __init__(self, input_dim, output_dim, K=4, use_bias=False, activation='elu'):
        super().__init__()
        self.K = K
        self.kernel = nn.Parameter(torch.empty(K * input_dim, output_dim))
        nn.init.xavier_uniform_(self.kernel)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = None

    def _normalize(self, A, eps=1e-6):
        A = A.clone()
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        A = A - torch.diag_embed(diag)
        D = torch.diag_embed(1.0 / (eps + A.sum(dim=2).sqrt()))
        return torch.bmm(torch.bmm(D, A), D)

    def forward(self, features, adj):
        L = self._normalize(adj)
        Xt = [features]
        Xt.append(torch.bmm(L, features))
        for k in range(2, self.K):
            Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])
        output = torch.cat(Xt, dim=-1)
        output = torch.matmul(output, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
