import os
import csv
import json
import gzip
import secrets

import numpy as np
import torch

from .utils import load_catalogue, load_FASTA, load_predicted_PDB, seq2onehot
from .torch_model import DeepFRIGCN, DeepCNN, LSTMLanguageModel


class PredictorPyTorch(object):
    def __init__(self, model_prefix, gcn=True, device='cpu'):
        self.model_prefix = model_prefix
        self.gcn = gcn
        self.device = torch.device(device)
        self._load_model()

    def _load_model(self):
        with open(self.model_prefix + '_model_params.json') as f:
            metadata = json.load(f)

        self.gonames = np.asarray(metadata['gonames'])
        self.goterms = np.asarray(metadata['goterms'])
        self.thresh = 0.1 * np.ones(len(self.goterms))

        output_dim = len(self.goterms)
        pt_dir = os.path.join(os.path.dirname(self.model_prefix), 'pytorch')
        model_basename = os.path.basename(self.model_prefix)

        if self.gcn:
            gc_dims = metadata.get('gc_dims', [512, 512, 512])
            fc_dims = metadata.get('fc_dims', [1024])
            gc_layer = metadata.get('gc_layer', 'MultiGraphConv')
            drop = metadata.get('dropout', 0.3)

            lm_model = None
            lm_name = metadata.get('lm_model_name')
            if lm_name:
                lm_model = LSTMLanguageModel(input_dim=26, hidden_dim=512)
                lm_pt = os.path.join(pt_dir, 'lstm_lm.pt')
                lm_sd = torch.load(lm_pt, map_location='cpu', weights_only=True)
                lm_model.load_state_dict(lm_sd)

            self.model = DeepFRIGCN(
                output_dim=output_dim, n_channels=26, gc_dims=gc_dims,
                fc_dims=fc_dims, drop=drop, gc_layer=gc_layer, lm_model=lm_model
            )
        else:
            num_filters = metadata.get('num_filters', [256] * 16)
            filter_lens = metadata.get('filter_lens',
                                       [8, 16, 25, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128])
            drop = metadata.get('dropout', 0.3)
            self.model = DeepCNN(
                output_dim=output_dim, n_channels=26,
                num_filters=num_filters, filter_lens=filter_lens, drop=drop
            )

        model_pt = os.path.join(pt_dir, f'{model_basename}.pt')
        sd = torch.load(model_pt, map_location='cpu', weights_only=True)
        if self.gcn and lm_model is not None:
            lm_sd_prefixed = {f'lm_model.{k}': v for k, v in lm_sd.items()}
            sd.update(lm_sd_prefixed)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _load_cmap(self, filename, cmap_thresh=10.0):
        if filename.endswith('.pdb'):
            D, seq = load_predicted_PDB(filename)
            A = np.double(D < cmap_thresh)
        elif filename.endswith('.npz'):
            cmap = np.load(filename)
            if 'C_alpha' not in cmap:
                raise ValueError("C_alpha not in *.npz dict.")
            D = cmap['C_alpha']
            A = np.double(D < cmap_thresh)
            seq = str(cmap['seqres'])
        elif filename.endswith('.pdb.gz'):
            rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
            with gzip.open(filename, 'rb') as f, open(rnd_fn, 'w') as out:
                out.write(f.read().decode())
            D, seq = load_predicted_PDB(rnd_fn)
            A = np.double(D < cmap_thresh)
            os.remove(rnd_fn)
        else:
            raise ValueError("File must be given in *.npz or *.pdb format.")
        S = seq2onehot(seq)
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)
        return A, S, seq

    @torch.no_grad()
    def predict(self, test_prot, cmap_thresh=10.0, chain='query_prot'):
        print("### Computing predictions on a single protein...")
        self.Y_hat = np.zeros((1, len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.test_prot_list = [chain]

        if self.gcn:
            A, S, seqres = self._load_cmap(test_prot, cmap_thresh=cmap_thresh)
            A_t = torch.tensor(A, dtype=torch.float32).to(self.device)
            S_t = torch.tensor(S, dtype=torch.float32).to(self.device)
            y = self.model(A_t, S_t).cpu().numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where(y >= self.thresh)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
        else:
            S = seq2onehot(str(test_prot))
            S = S.reshape(1, *S.shape)
            S_t = torch.tensor(S, dtype=torch.float32).to(self.device)
            y = self.model(S_t).cpu().numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], test_prot]
            go_idx = np.where(y >= self.thresh)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    @torch.no_grad()
    def predict_from_fasta(self, fasta_fn):
        print("### Computing predictions from fasta...")
        self.test_prot_list, sequences = load_FASTA(fasta_fn)
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}

        for i, chain in enumerate(self.test_prot_list):
            S = seq2onehot(str(sequences[i]))
            S = S.reshape(1, *S.shape)
            S_t = torch.tensor(S, dtype=torch.float32).to(self.device)
            y = self.model(S_t).cpu().numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], str(sequences[i])]
            go_idx = np.where(y >= self.thresh)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def export_csv(self, output_fn, verbose):
        with open(output_fn, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI (PyTorch).'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    if verbose:
                        print(prot, row[0], '{:.5f}'.format(row[2]), row[1])
                    writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])

    def save_predictions(self, output_fn):
        print("### Saving predictions to *.json file...")
        with open(output_fn, 'w') as fw:
            out_data = {'pdb_chains': self.test_prot_list,
                        'Y_hat': self.Y_hat.tolist(),
                        'goterms': self.goterms.tolist(),
                        'gonames': self.gonames.tolist()}
            json.dump(out_data, fw, indent=1)
