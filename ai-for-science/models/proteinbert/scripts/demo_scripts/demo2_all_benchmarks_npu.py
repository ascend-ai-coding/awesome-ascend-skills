import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from scipy.stats import spearmanr
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from proteinbert_pytorch.convert_weights import convert_tf_to_pytorch
from proteinbert_pytorch.inference import tokenize_seqs

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protein_benchmarks')
PKL_PATH = os.path.expanduser('~/proteinbert_models/epoch_92400_sample_23500000.pkl')
DEVICE = 'npu:0'

def log(msg):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)

BENCHMARKS = [
    ('signalP_binary', 'binary'),
    ('fluorescence', 'numeric'),
    ('remote_homology', 'categorical'),
    ('stability', 'numeric'),
    ('ProFET_NP_SP_Cleaved', 'binary'),
]

settings = dict(max_dataset_size=None, max_epochs_per_stage=40, seq_len=512,
    batch_size=32, final_epoch_seq_len=1024,
    initial_lr_with_frozen_pretrained_layers=1e-02,
    initial_lr_with_all_layers=1e-04, final_epoch_lr=1e-05, dropout_rate=0.5)

class ProteinDS(Dataset):
    def __init__(self, toks, anns, lbls):
        self.toks=torch.from_numpy(toks).long(); self.anns=torch.from_numpy(anns).float(); self.lbls=torch.from_numpy(lbls).float()
    def __len__(self): return len(self.lbls)
    def __getitem__(self,i): return self.toks[i],self.anns[i],self.lbls[i]

class FTModel(nn.Module):
    def __init__(self, base, task='binary', n_cls=1, dropout=0.5):
        super().__init__()
        self.base=base; self.task=task; self.drop=nn.Dropout(dropout)
        global_concat_dim = 13 * base.d_hidden_global + base.n_annotations
        out_dim = n_cls if task=='categorical' else 1
        self.head = nn.Linear(global_concat_dim, out_dim)
    def forward(self, seq, ann):
        h_s=self.base.seq_embedding(seq); h_g=F.gelu(self.base.global_input_dense(ann))
        global_outputs=[h_g]
        for block in self.base.blocks:
            seqed_global=F.gelu(block.global_to_seq_dense(h_g)).unsqueeze(1)
            seq_t=h_s.transpose(1,2)
            narrow=F.gelu(block.narrow_conv(seq_t)).transpose(1,2)
            wide=F.gelu(block.wide_conv(seq_t)).transpose(1,2)
            h_s=block.seq_norm1(h_s+seqed_global+narrow+wide)
            h_s=block.seq_norm2(h_s+F.gelu(block.seq_dense(h_s)))
            dense_g=F.gelu(block.global_dense1(h_g)); attn=block.global_attention(h_g,h_s)
            h_g=block.global_norm1(h_g+dense_g+attn); global_outputs.append(h_g)
            h_g=block.global_norm2(h_g+F.gelu(block.global_dense2(h_g))); global_outputs.append(h_g)
        out_ann=torch.sigmoid(self.base.output_annotations_dense(h_g)); global_outputs.append(out_ann)
        concat_global=torch.cat(global_outputs,dim=-1)
        lo=self.head(self.drop(concat_global))
        return lo if self.task=='categorical' else lo.squeeze(-1)

def mkdl(seqs,labels,n_ann,sl,bs,shuf=False):
    t=tokenize_seqs(list(seqs),sl); a=np.zeros((len(seqs),n_ann),dtype=np.float32)
    l=np.array(list(labels),dtype=np.float32)
    return DataLoader(ProteinDS(t,a,l),batch_size=bs,shuffle=shuf,num_workers=0)

def loss_fn(lo,lb,task):
    if task=='binary': return F.binary_cross_entropy_with_logits(lo,lb)
    elif task=='categorical': return F.cross_entropy(lo,lb.long())
    else: return F.mse_loss(lo,lb)

def train_ep(m,dl,opt,dev,task):
    m.train(); tl,n=0.,0
    for tk,an,lb in dl:
        tk,an,lb=tk.to(dev),an.to(dev),lb.to(dev); opt.zero_grad()
        loss=loss_fn(m(tk,an),lb,task); loss.backward(); opt.step()
        tl+=loss.item()*len(lb); n+=len(lb)
    return tl/n

def eval_fn(m,dl,dev,task):
    m.eval(); ps,ls,tl,n=[],[],0.,0
    with torch.no_grad():
        for tk,an,lb in dl:
            tk,an,lb=tk.to(dev),an.to(dev),lb.to(dev)
            lo=m(tk,an); loss=loss_fn(lo,lb,task)
            if task=='binary': ps.append(torch.sigmoid(lo).cpu().numpy())
            elif task=='categorical': ps.append(lo.argmax(-1).float().cpu().numpy())
            else: ps.append(lo.cpu().numpy())
            ls.append(lb.cpu().numpy()); tl+=loss.item()*len(lb); n+=len(lb)
    return tl/n,np.concatenate(ps),np.concatenate(ls)

def run_stage(ft,trdl,vdl,dev,task,epochs,lr,patience,freeze=False):
    if freeze:
        for p in ft.base.parameters(): p.requires_grad=False
        ft.head.requires_grad_(True); ft.drop.requires_grad_(True)
    else:
        for p in ft.parameters(): p.requires_grad=True
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,ft.parameters()),lr=lr)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=1,factor=0.25,min_lr=1e-5)
    best,bst,w=float('inf'),None,0
    for ep in range(1,epochs+1):
        tl=train_ep(ft,trdl,opt,dev,task); vl,vp,vy=eval_fn(ft,vdl,dev,task); sch.step(vl)
        tag=' [frozen]' if freeze else ''; extra=''
        if task=='binary':
            try: extra=f', auc: {roc_auc_score(vy,vp):.4f}'
            except: pass
        print(f'  Epoch {ep}/{epochs}{tag} - train: {tl:.4f}, val: {vl:.4f}, lr: {opt.param_groups[0]["lr"]:.4e}{extra}')
        if vl<best: best=vl; bst={k:v.clone() for k,v in ft.state_dict().items()}; w=0
        else:
            w+=1
            if w>=patience: print(f'  Early stopping at epoch {ep}'); break
    if bst: ft.load_state_dict(bst)

def load_benchmark_dataset(name):
    tr=pd.read_csv(os.path.join(BENCHMARKS_DIR,f'{name}.train.csv')).dropna().drop_duplicates()
    te=pd.read_csv(os.path.join(BENCHMARKS_DIR,f'{name}.test.csv')).dropna().drop_duplicates()
    vp=os.path.join(BENCHMARKS_DIR,f'{name}.valid.csv')
    if os.path.exists(vp): va=pd.read_csv(vp).dropna().drop_duplicates()
    else:
        log(f'Validation set {os.path.join(BENCHMARKS_DIR, f"{name}.valid.csv")} missing. Splitting training set instead.')
        tr,va=train_test_split(tr,stratify=tr['label'],test_size=0.1,random_state=0)
    return tr,va,te

def evaluate_by_len_pt(ft_model, seqs, labels, n_ann, device, task,
                       start_seq_len=512, start_batch_size=32, increase_factor=2):
    """对标 TF 版 evaluate_by_len: 按序列长度分组评估"""
    df = pd.DataFrame({'seq': list(seqs), 'label': list(labels)})
    df['seq_len'] = df['seq'].str.len() + 2
    results_rows = []
    all_preds, all_labels = [], []
    seq_len = start_seq_len
    batch_size = start_batch_size
    prev_max = 0
    while True:
        subset = df[(df['seq_len'] <= seq_len) & (df['seq_len'] > prev_max)]
        if len(subset) == 0:
            if seq_len >= df['seq_len'].max():
                break
            prev_max = seq_len
            seq_len *= increase_factor
            batch_size = max(batch_size // increase_factor, 1)
            continue
        dl = mkdl(subset['seq'], subset['label'], n_ann, seq_len, batch_size)
        _, preds, lbls = eval_fn(ft_model, dl, device, task)
        row = {'Model seq len': seq_len, '# records': len(subset)}
        if task == 'binary':
            try: row['AUC'] = roc_auc_score(lbls, preds)
            except: row['AUC'] = float('nan')
        elif task == 'numeric':
            row["Spearman's rank correlation"] = spearmanr(lbls, preds).correlation
        elif task == 'categorical':
            row['Accuracy'] = accuracy_score(lbls, preds)
        results_rows.append(row)
        all_preds.append(preds)
        all_labels.append(lbls)
        prev_max = seq_len
        if seq_len >= df['seq_len'].max():
            break
        seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    row_all = {'Model seq len': 'All', '# records': len(all_labels)}
    if task == 'binary':
        try: row_all['AUC'] = roc_auc_score(all_labels, all_preds)
        except: row_all['AUC'] = float('nan')
    elif task == 'numeric':
        row_all["Spearman's rank correlation"] = spearmanr(all_labels, all_preds).correlation
    elif task == 'categorical':
        row_all['Accuracy'] = accuracy_score(all_labels, all_preds)
    results_rows.append(row_all)
    results_df = pd.DataFrame(results_rows).set_index('Model seq len')
    if task == 'binary':
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = pd.DataFrame(sk_cm(all_labels, (all_preds > 0.5).astype(int)))
    else:
        cm = None
    return results_df, cm

def run_benchmark(bname, task, base_model, n_ann):
    log(f'========== {bname} ==========')
    log(f'Output type: {task}')
    tr,va,te = load_benchmark_dataset(bname)
    log(f'{len(tr)} training set records, {len(va)} validation set records, {len(te)} test set records.')
    if settings['max_dataset_size'] is not None:
        mx=settings['max_dataset_size']
        tr=tr.sample(min(mx,len(tr)),random_state=0)
        va=va.sample(min(mx,len(va)),random_state=0)
        te=te.sample(min(mx,len(te)),random_state=0)
    if task=='categorical':
        for d in [tr,va,te]: d['label']=d['label'].astype(str)
        unique=sorted(set(tr['label'])|set(va['label'])|set(te['label']))
        lbl_map={l:i for i,l in enumerate(unique)}; n_cls=len(unique)
        for d in [tr,va,te]: d['label']=d['label'].map(lbl_map).astype(float)
        log(f'{n_cls} unique labels.')
    elif task=='binary': n_cls=1
    else:
        for d in [tr,va,te]: d['label']=d['label'].astype(float)
        n_cls=1
    sl=settings['seq_len']; bs=settings['batch_size']
    ft=FTModel(base_model,task,n_cls,settings['dropout_rate']).to(DEVICE)
    trdl=mkdl(tr['seq'],tr['label'],n_ann,sl,bs,True); vdl=mkdl(va['seq'],va['label'],n_ann,sl,bs)
    log('Training with frozen pretrained layers...')
    run_stage(ft,trdl,vdl,DEVICE,task,settings['max_epochs_per_stage'],
              settings['initial_lr_with_frozen_pretrained_layers'],2,freeze=True)
    log('Training the entire fine-tuned model...')
    run_stage(ft,trdl,vdl,DEVICE,task,settings['max_epochs_per_stage'],
              settings['initial_lr_with_all_layers'],2,freeze=False)
    fsl=settings['final_epoch_seq_len']; fbs=max(int(bs/(fsl/sl)),1)
    log(f'Training on final epochs of sequence length {fsl}...')
    trdl_f=mkdl(tr['seq'],tr['label'],n_ann,fsl,fbs,True); vdl_f=mkdl(va['seq'],va['label'],n_ann,fsl,fbs)
    run_stage(ft,trdl_f,vdl_f,DEVICE,task,1,settings['final_epoch_lr'],2,freeze=False)
    for dsname,ds in [('Training-set',tr),('Validation-set',va),('Test-set',te)]:
        log(f'*** {dsname} performance: ***')
        results, cm = evaluate_by_len_pt(ft, ds['seq'], ds['label'], n_ann, DEVICE, task,
                                         start_seq_len=sl, start_batch_size=bs)
        print(results.to_string())
        if cm is not None:
            log('Confusion matrix:')
            print(cm.to_string())
    log(f'[Compare Log] {bname} done.')
    del ft

pretrained_model, n_ann = convert_tf_to_pytorch(PKL_PATH)
pretrained_model = pretrained_model.to(DEVICE)
for bname, task in BENCHMARKS:
    run_benchmark(bname, task, pretrained_model, n_ann)
log('Done.')
