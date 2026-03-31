import os
import json
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from deepfrier.torch_predictor import PredictorPyTorch
from deepfrier.utils import seq2onehot

REFERENCE = {
    "cnn_mf_seq": {"GO:0005509": 0.99769},
    "cnn_mf_fasta": {
        "1S3P-A": {"GO:0005509": 0.99769},
        "2J9H-A": {"GO:0004364": 0.46937, "GO:0016765": 0.19910},
    },
}

def compare_score(name, expected, actual, tol=1e-4):
    diff = abs(expected - actual)
    status = "PASS" if diff < tol else ("WARN" if diff < 1e-2 else "FAIL")
    print(f"  [{status}] {name:45s} expected={expected:.5f} actual={actual:.5f} diff={diff:.6f}")
    return status

def main():
    config = json.load(open('trained_models/model_config.json'))
    test_seq = 'SMTDLLSAEDIKKAIGAFTAADSFDHKKFFQMVGLKKKSADDVKKVFHILDKDKDGFIDEDELGSILKGFSSDARDLSAKETKTLMAAGDKDGDGKIGVEEFSTLVAES'

    print("=" * 75)
    print("DeepFRI NPU (PyTorch) vs README Accuracy Comparison")
    print("=" * 75)

    print("\n--- Option 2: CNN sequence input (mf) ---")
    print("README reference: query_prot GO:0005509 0.99769 calcium ion binding")
    predictor = PredictorPyTorch(config['cnn']['models']['mf'], gcn=False, device='npu:0')
    predictor.predict(test_seq)
    for row in sorted(predictor.prot2goterms['query_prot'], key=lambda x: x[2], reverse=True)[:3]:
        expected = REFERENCE["cnn_mf_seq"].get(row[0])
        if expected:
            compare_score(f"query_prot {row[0]} {row[1]}", expected, row[2])
        else:
            print(f"  [INFO] query_prot {row[0]:15s} score={row[2]:.5f} {row[1]}")

    print("\n--- Option 3: CNN fasta input (mf) ---")
    print("README reference: 1S3P-A GO:0005509 0.99769 / 2J9H-A GO:0004364 0.46937")
    predictor2 = PredictorPyTorch(config['cnn']['models']['mf'], gcn=False, device='npu:0')
    predictor2.predict_from_fasta('examples/pdb_chains.fasta')
    for chain in predictor2.test_prot_list:
        ref = REFERENCE["cnn_mf_fasta"].get(chain, {})
        for row in sorted(predictor2.prot2goterms.get(chain, []), key=lambda x: x[2], reverse=True)[:3]:
            expected = ref.get(row[0])
            if expected:
                compare_score(f"{chain} {row[0]} {row[1]}", expected, row[2])
            else:
                print(f"  [INFO] {chain} {row[0]:15s} score={row[2]:.5f} {row[1]}")

    print("\n--- CNN all ontologies (sequence input) ---")
    for ont in ['mf', 'bp', 'cc', 'ec']:
        pred = PredictorPyTorch(config['cnn']['models'][ont], gcn=False, device='npu:0')
        pred.predict(test_seq)
        n_pred = len(pred.prot2goterms.get('query_prot', []))
        top = sorted(pred.prot2goterms.get('query_prot', []), key=lambda x: x[2], reverse=True)
        if top:
            print(f"  [OK] {ont.upper():3s} top: {top[0][0]} score={top[0][2]:.5f} ({top[0][1]}) [{n_pred} predictions]")
        else:
            print(f"  [OK] {ont.upper():3s} no predictions above threshold (expected for some proteins)")

    print("\n" + "=" * 75)
    print("Verification complete.")
    print("=" * 75)

if __name__ == '__main__':
    main()
