import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import json
import numpy as np
import h5py
import torch
from collections import OrderedDict


def load_h5_weights(h5_path):
    weights = {}
    f = h5py.File(h5_path, 'r')
    mw = f['model_weights']
    for layer_name in mw.keys():
        grp = mw[layer_name]
        layer_weights = {}
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                layer_weights[name] = np.array(obj)
        grp.visititems(collect)
        if layer_weights:
            weights[layer_name] = layer_weights
    f.close()
    return weights


def convert_lstm_weights(tf_weights):
    state_dict = OrderedDict()

    lstm1_kernel = tf_weights['LSTM1']['LSTM1/kernel:0']
    lstm1_rec = tf_weights['LSTM1']['LSTM1/recurrent_kernel:0']
    lstm1_bias = tf_weights['LSTM1']['LSTM1/bias:0']

    lstm2_kernel = tf_weights['LSTM2']['LSTM2/kernel:0']
    lstm2_rec = tf_weights['LSTM2']['LSTM2/recurrent_kernel:0']
    lstm2_bias = tf_weights['LSTM2']['LSTM2/bias:0']

    def convert_cudnnlstm(kernel, rec_kernel, bias, hidden_dim):
        return kernel.T, rec_kernel.T, bias[:4 * hidden_dim], bias[4 * hidden_dim:]

    hidden = 512
    w_ih1, w_hh1, b_ih1, b_hh1 = convert_cudnnlstm(lstm1_kernel, lstm1_rec, lstm1_bias, hidden)
    state_dict['lstm1.weight_ih_l0'] = torch.tensor(w_ih1)
    state_dict['lstm1.weight_hh_l0'] = torch.tensor(w_hh1)
    state_dict['lstm1.bias_ih_l0'] = torch.tensor(b_ih1)
    state_dict['lstm1.bias_hh_l0'] = torch.tensor(b_hh1)

    w_ih2, w_hh2, b_ih2, b_hh2 = convert_cudnnlstm(lstm2_kernel, lstm2_rec, lstm2_bias, hidden)
    state_dict['lstm2.weight_ih_l0'] = torch.tensor(w_ih2)
    state_dict['lstm2.weight_hh_l0'] = torch.tensor(w_hh2)
    state_dict['lstm2.bias_ih_l0'] = torch.tensor(b_ih2)
    state_dict['lstm2.bias_hh_l0'] = torch.tensor(b_hh2)

    return state_dict


def convert_gcn_weights(tf_weights, num_gc=3):
    state_dict = OrderedDict()

    aa_kernel = tf_weights['AA_embedding']['AA_embedding/kernel:0']
    state_dict['aa_embedding.weight'] = torch.tensor(aa_kernel.T)

    if 'LM_embedding' in tf_weights:
        lm_kernel = tf_weights['LM_embedding']['LM_embedding/kernel:0']
        lm_bias = tf_weights['LM_embedding']['LM_embedding/bias:0']
        state_dict['lm_embedding.weight'] = torch.tensor(lm_kernel.T)
        state_dict['lm_embedding.bias'] = torch.tensor(lm_bias)

    gc_prefix_map = {0: 'multi_graph_conv', 1: 'multi_graph_conv_1', 2: 'multi_graph_conv_2'}
    for i in range(num_gc):
        tf_name = gc_prefix_map[i]
        state_dict[f'gc_layers.{i}.kernel'] = torch.tensor(tf_weights[tf_name][f'{tf_name}/kernel:0'])
        bias_key = f'{tf_name}/bias:0'
        if bias_key in tf_weights[tf_name]:
            state_dict[f'gc_layers.{i}.bias'] = torch.tensor(tf_weights[tf_name][bias_key])

    fc_kernel = tf_weights['dense']['dense/kernel:0']
    fc_bias = tf_weights['dense']['dense/bias:0']
    state_dict['fc_layers.0.weight'] = torch.tensor(fc_kernel.T)
    state_dict['fc_layers.0.bias'] = torch.tensor(fc_bias)

    label_kernel = tf_weights['labels']['labels/dense_1/kernel:0']
    label_bias = tf_weights['labels']['labels/dense_1/bias:0']
    state_dict['func_predictor.dense.weight'] = torch.tensor(label_kernel.T)
    state_dict['func_predictor.dense.bias'] = torch.tensor(label_bias)

    return state_dict


def convert_cnn_weights(tf_weights):
    state_dict = OrderedDict()

    conv_keys = sorted([k for k in tf_weights.keys() if k.startswith('conv1d')],
                       key=lambda x: int(x.split('_')[-1]) if '_' in x else -1)

    for i, ck in enumerate(conv_keys):
        tf_kernel = tf_weights[ck][f'{ck}/kernel:0']
        tf_bias = tf_weights[ck][f'{ck}/bias:0']
        state_dict[f'conv_layers.{i}.weight'] = torch.tensor(np.transpose(tf_kernel, (2, 1, 0)))
        state_dict[f'conv_layers.{i}.bias'] = torch.tensor(tf_bias)

    bn = tf_weights['batch_normalization']
    state_dict['bn.weight'] = torch.tensor(bn['batch_normalization/gamma:0'])
    state_dict['bn.bias'] = torch.tensor(bn['batch_normalization/beta:0'])
    state_dict['bn.running_mean'] = torch.tensor(bn['batch_normalization/moving_mean:0'])
    state_dict['bn.running_var'] = torch.tensor(bn['batch_normalization/moving_variance:0'])

    label_kernel = tf_weights['labels']['labels/dense/kernel:0']
    label_bias = tf_weights['labels']['labels/dense/bias:0']
    state_dict['func_predictor.dense.weight'] = torch.tensor(label_kernel.T)
    state_dict['func_predictor.dense.bias'] = torch.tensor(label_bias)

    return state_dict


def convert_all_models(trained_dir='trained_models', output_dir='trained_models/pytorch'):
    os.makedirs(output_dir, exist_ok=True)

    print("Converting LSTM LM weights...")
    tf_w = load_h5_weights(os.path.join(trained_dir, 'lstm_lm.hdf5'))
    sd = convert_lstm_weights(tf_w)
    torch.save(sd, os.path.join(output_dir, 'lstm_lm.pt'))
    print(f"  Saved {len(sd)} tensors")

    config = json.load(open(os.path.join(trained_dir, 'model_config.json')))

    for ont, prefix in config['gcn']['models'].items():
        print(f"\nConverting GCN model: {ont} ...")
        tf_w = load_h5_weights(prefix + '.hdf5')
        params = json.load(open(prefix + '_model_params.json'))
        sd = convert_gcn_weights(tf_w, num_gc=len(params['gc_dims']))
        torch.save(sd, os.path.join(output_dir, f'{os.path.basename(prefix)}.pt'))
        print(f"  Saved {len(sd)} tensors")

    for ont, prefix in config['cnn']['models'].items():
        print(f"\nConverting CNN model: {ont} ...")
        tf_w = load_h5_weights(prefix + '.hdf5')
        sd = convert_cnn_weights(tf_w)
        torch.save(sd, os.path.join(output_dir, f'{os.path.basename(prefix)}.pt'))
        print(f"  Saved {len(sd)} tensors")

    print("\nAll models converted successfully!")


if __name__ == '__main__':
    convert_all_models()
