import json
import argparse
import torch_npu
from torch_npu.contrib import transfer_to_npu
from deepfrier.torch_predictor import PredictorPyTorch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str, help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str, help="Protein contact map (*.npz).")
    parser.add_argument('-pdb', '--pdb_fn', type=str, help="Protein PDB file.")
    parser.add_argument('--fasta_fn', type=str, help="Fasta file with protein sequences.")
    parser.add_argument('--model_config', type=str, default='./trained_models/model_config.json',
                        help="JSON file with model names.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True,
                        choices=['mf', 'bp', 'cc', 'ec'], help="Ontology.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI_NPU',
                        help="Save predictions in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--device', type=str, default='npu:0', help="Device (npu:0, cpu).")
    args = parser.parse_args()

    with open(args.model_config) as json_file:
        params = json.load(json_file)

    if args.seq is not None or args.fasta_fn is not None:
        params = params['cnn']
    elif args.cmap is not None or args.pdb_fn is not None:
        params = params['gcn']
    gcn = params['gcn']
    models = params['models']

    for ont in args.ontology:
        predictor = PredictorPyTorch(models[ont], gcn=gcn, device=args.device)
        if args.seq is not None:
            predictor.predict(args.seq)
        if args.cmap is not None:
            predictor.predict(args.cmap)
        if args.pdb_fn is not None:
            predictor.predict(args.pdb_fn)
        if args.fasta_fn is not None:
            predictor.predict_from_fasta(args.fasta_fn)

        predictor.export_csv(args.output_fn_prefix + "_" + ont.upper() + "_predictions.csv", args.verbose)
        predictor.save_predictions(args.output_fn_prefix + "_" + ont.upper() + "_pred_scores.json")
