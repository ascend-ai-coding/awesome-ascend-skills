import pickle
import numpy as np
import torch


def convert_tf_to_pytorch(pkl_path, output_path=None):
    with open(pkl_path, 'rb') as f:
        n_annotations, model_weights, optimizer_weights = pickle.load(f)

    print(f'Loaded TF weights: n_annotations={n_annotations}, '
          f'num weight arrays={len(model_weights)}')

    from .model import ProteinBERTModel
    n_tokens = 26
    model = ProteinBERTModel(vocab_size=n_tokens, n_annotations=n_annotations)

    weight_map = _build_weight_map(model_weights, n_tokens, n_annotations, model)

    state_dict = model.state_dict()
    for pt_key, np_val in weight_map.items():
        tensor_val = torch.from_numpy(np_val)
        if state_dict[pt_key].shape != tensor_val.shape:
            raise ValueError(f'Shape mismatch for {pt_key}: '
                             f'expected {state_dict[pt_key].shape}, '
                             f'got {tensor_val.shape}')
        state_dict[pt_key] = tensor_val

    model.load_state_dict(state_dict)
    print('Successfully loaded all TF weights into PyTorch model.')

    if output_path:
        torch.save({
            'n_annotations': n_annotations,
            'state_dict': model.state_dict(),
        }, output_path)
        print(f'Saved PyTorch model to {output_path}')

    return model, n_annotations


def _build_weight_map(tf_weights, vocab_size, n_annotations, model):
    weight_map = {}
    idx = 0

    dense_w = tf_weights[idx]; idx += 1
    dense_b = tf_weights[idx]; idx += 1
    weight_map['global_input_dense.weight'] = dense_w.T
    weight_map['global_input_dense.bias'] = dense_b

    emb_w = tf_weights[idx]; idx += 1
    weight_map['seq_embedding.weight'] = emb_w

    for block_idx in range(model.n_blocks):
        prefix = f'blocks.{block_idx}'

        w = tf_weights[idx]; idx += 1
        b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.global_to_seq_dense.weight'] = w.T
        weight_map[f'{prefix}.global_to_seq_dense.bias'] = b

        w = tf_weights[idx]; idx += 1
        b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.narrow_conv.weight'] = np.transpose(w, (2, 1, 0))
        weight_map[f'{prefix}.narrow_conv.bias'] = b

        w = tf_weights[idx]; idx += 1
        b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.wide_conv.weight'] = np.transpose(w, (2, 1, 0))
        weight_map[f'{prefix}.wide_conv.bias'] = b

        ln_g = tf_weights[idx]; idx += 1
        ln_b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.seq_norm1.weight'] = ln_g
        weight_map[f'{prefix}.seq_norm1.bias'] = ln_b

        w = tf_weights[idx]; idx += 1
        b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.seq_dense.weight'] = w.T
        weight_map[f'{prefix}.seq_dense.bias'] = b

        ln_g = tf_weights[idx]; idx += 1
        ln_b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.seq_norm2.weight'] = ln_g
        weight_map[f'{prefix}.seq_norm2.bias'] = ln_b

        w = tf_weights[idx]; idx += 1
        b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.global_dense1.weight'] = w.T
        weight_map[f'{prefix}.global_dense1.bias'] = b

        Wq = tf_weights[idx]; idx += 1
        Wk = tf_weights[idx]; idx += 1
        Wv = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.global_attention.Wq'] = Wq
        weight_map[f'{prefix}.global_attention.Wk'] = Wk
        weight_map[f'{prefix}.global_attention.Wv'] = Wv

        ln_g = tf_weights[idx]; idx += 1
        ln_b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.global_norm1.weight'] = ln_g
        weight_map[f'{prefix}.global_norm1.bias'] = ln_b

        w = tf_weights[idx]; idx += 1
        b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.global_dense2.weight'] = w.T
        weight_map[f'{prefix}.global_dense2.bias'] = b

        ln_g = tf_weights[idx]; idx += 1
        ln_b = tf_weights[idx]; idx += 1
        weight_map[f'{prefix}.global_norm2.weight'] = ln_g
        weight_map[f'{prefix}.global_norm2.bias'] = ln_b

    w = tf_weights[idx]; idx += 1
    b = tf_weights[idx]; idx += 1
    weight_map['output_seq_dense.weight'] = w.T
    weight_map['output_seq_dense.bias'] = b

    w = tf_weights[idx]; idx += 1
    b = tf_weights[idx]; idx += 1
    weight_map['output_annotations_dense.weight'] = w.T
    weight_map['output_annotations_dense.bias'] = b

    print(f'Mapped {idx} TF weight arrays to PyTorch state dict '
          f'({len(weight_map)} entries)')
    return weight_map
