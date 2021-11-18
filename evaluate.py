from argparse import ArgumentParser
import os

import torch
from tqdm.auto import tqdm

from training_pipeline import calculate_epoch_metrics, do_phase, get_dataloaders, initialize_model
from models import *

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--model_file", type=str, required=True)
    psr.add_argument("--model_name", type=str, required=True)
    psr.add_argument("--dataset", type=str, required=True, choices=['mnist', 'camelyon17', 'iwildcam'])
    psr.add_argument("--corr", type=float, required=True)
    psr.add_argument("--batch_size", type=int, default=32)
    psr.add_argument("--num_workers", type=int, default=os.cpu_count() // 2)
    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--root_dir", type=str, default='/scratch/eecs542f21_class_root/eecs542f21_class/shared_data/dssr_datasets/WildsData/camelyon17_v1.0')
    psr.add_argument("--no-state-dict", action='store_true')
    args = psr.parse_args()

    print("Loading model...")
    model, _ = initialize_model(
         args.model_name,
         2, # num classes is always 2
         True, # set this feature extraction flag to true to freeze stuff
         use_pretrained=False
    )
    if not args.no_state_dict:
        weights = torch.load(args.model_file)
    else: # need to extract the state dict from the model file
        weights = torch.load(args.model_file).state_dict() 
    model.load_state_dict(weights)
    model.eval()

    print("Loading dataset...")
    _, test_dl = get_dataloaders(
        args.dataset,
        args.root_dir,
        args.corr,
        args.seed,
        args.batch_size,
        args.num_workers,
        test_only=True,
    )

    pbar = tqdm(enumerate(test_dl), total=len(test_dl))
    _, _, all_y, all_preds, all_scores, _, all_domains, all_domain_preds, all_domain_scores = do_phase('val', model, pbar)
    _, acc_cls, f1_cls, auc_cls = calculate_epoch_metrics(0, all_y, all_preds, all_scores)
    _, acc_dom, f1_dom, auc_dom = calculate_epoch_metrics(0, all_domains, all_domain_preds, all_domain_scores)
    print('C-Acc: {:.4f} C-F1: {:.4f} C-AUC: {:.4f}'.format(acc_cls, f1_cls, auc_cls))
    print('D-Acc: {:.4f} D-F1: {:.4f} D-AUC: {:.4f}'.format(acc_dom, f1_dom, auc_dom))
    
