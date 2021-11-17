from argparse import ArgumentParser
import os

import torch

from training_pipeline import do_phase, get_dataloaders, initialize_model

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--model-file", type=str, required=True)
    psr.add_argument("--model-type", type=str, required=True)
    psr.add_argument("--dataset", type=str, required=True, choices=['mnist', 'camelyon17', 'iwildcam'])
    psr.add_argument("--corr", type=float, required=True)
    psr.add_argument("--batch-size", type=int, default=32)
    psr.add_argument("--num-workers", type=int, default=os.cpu_count() // 2)
    psr.add_argument("--seed", type=int, default=42)
    # psr.add_argument("--no-state-dict", action='store_true')
    args = psr.parse_args()

    print("Loading model...")
    
    # if not args.no_state_dict:
    #model = initialize_model(
    #    args.model_type,
    #    2, # num classes is always 2
    #    True, # set this feature extraction flag to true to freeze stuff
    #    use_pretrained=False
    #)
    # else:
    model = torch.load(args.model_file)
    model.eval()

    print("Loading dataset...")
    _, test_dl = get_dataloaders(
        args.dataset_name,
        root_dir,
        args.corr,
        args.seed,
        args.batch_size,
        args.num_workers,
        test_only=True,
    )

    _, _, all_y, all_preds, all_scores, _, all_domains, all_domain_preds, all_domain_scores = do_phase(phase, model, pbar)
    _, acc_cls, f1_cls, auc_cls = calculate_epoch_metrics(running_loss_class, all_y, all_preds, all_scores)
    _, acc_dom, f1_dom, auc_dom = calculate_epoch_metrics(running_loss_domain, all_domains, all_domain_preds, all_domain_scores)
    print('C-Acc: {:.4f} C-F1: {:.4f} C-AUC: {:.4f}'.format(acc_cls, f1_cls, auc_cls))
    print('D-Acc: {:.4f} D-F1: {:.4f} D-AUC: {:.4f}'.format(acc_dom, f1_dom, auc_dom))
    
