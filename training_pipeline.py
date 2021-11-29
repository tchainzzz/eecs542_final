from __future__ import division, print_function

import copy
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import datasets
import torchvision
# from torchvision import models, transforms
from models import *
from torchvision import transforms
from utils import parse_argdict

# Logging
import wandb

#note: originally designed for resnet but should work on anything that can use sequential (for first iteration of this implementation)
# class TwoHeadResNet(torch.nn.Module):
#     def __init__(self, resnetModel):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(TwoHeadResNet, self).__init__()
#         self.l_input_size = resnetModel.fc.in_features
#         self.resnetBackbone = torch.nn.Sequential(*(list(resnetModel.children())[:-1]))

#         self.classHead = torch.nn.Linear(self.l_input_size, 1)
#         self.domainHead = torch.nn.Linear(self.l_input_size, 1)

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         backboneOut = self.resnetBackbone(x)
#         backboneOut = backboneOut.view(-1, self.l_input_size)
#         classOut = self.classHead(backboneOut)
#         domainOut = self.domainHead(backboneOut)
#         return torch.sigmoid(classOut), torch.sigmoid(domainOut)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)

def do_phase(phase, model, pbar, criterion=None, optimizer=None, limit_batches=-1):
    if phase == 'train':
        model.train()  # Set model to training mode
        assert optimizer is not None, "You need to have an optimizer to, uh, optimize stuff."
    else:
        model.eval()   # Set model to evaluate mode

    # initialize metric-storing structures
    running_loss_class = 0.0
    running_loss_domain = 0.0

    all_y = torch.Tensor().to(device)
    all_scores = torch.Tensor().to(device)
    all_preds = torch.Tensor().to(device)

    all_domains = torch.Tensor().to(device)
    all_domain_scores = torch.Tensor().to(device)
    all_domain_preds = torch.Tensor().to(device)

    # create progress bar
    for i, (inputs, labels, domains) in pbar:
        # iterate through data -- batch optimization
        if i == limit_batches:
            break

        # move data to correct devices/shape as needed
        inputs = inputs.to(device)
        labels = labels.to(device)
        domains = domains.to(device)

        labels = labels.view(-1, 1)
        labels = labels.to(torch.float32)
        domains = domains.view(-1, 1)
        domains = domains.to(torch.float32)

        if phase == 'train':
            # zero the parameter gradients
            optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
            scores_class, scores_domain = model(inputs) # list of probabilities with shape (1, batch_size)
            if criterion:
                loss_class = criterion(scores_class, labels)
                loss_domain = criterion(scores_domain, domains)

            preds_class = torch.where(scores_class < 0.5, 0, 1)
            preds_domain = torch.where(scores_domain < 0.5, 0, 1)
            # backward + optimize only if in training phase
            if phase == 'train':
                loss_class.backward()
                loss_domain.backward()
                optimizer.step()

        all_y = torch.cat((all_y, labels))
        all_preds = torch.cat((all_preds, preds_class))
        all_scores = torch.cat((all_scores, scores_class))

        all_domains = torch.cat((all_domains, domains))
        all_domain_preds = torch.cat((all_domain_preds, preds_domain))
        all_domain_scores = torch.cat((all_domain_scores, scores_domain))

        # statistics
        if criterion:
            running_loss_class += loss_class.item() * inputs.size(0)
            running_loss_domain += loss_domain.item() * inputs.size(0)
        pbar.set_postfix({
            "loss": running_loss_class / all_y.size(0),
            "acc": accuracy_score(all_y.detach().cpu().numpy(), all_preds.detach().cpu().numpy()),
            "f1": f1_score(all_y.detach().cpu().numpy(), all_preds.detach().cpu().numpy()),
            "auc": roc_auc_score(all_y.detach().cpu().numpy(), all_scores.detach().cpu().numpy())
        })

        if phase == 'train':
            # Logging: 
            wandb.log({
                phase+'_loss_classification_step':loss_class.item(),
                phase+'_loss_domain_step':loss_domain.item(), 
                phase+'_accuracy_classification_step':accuracy_score(labels.detach().cpu().numpy(),preds_class.detach().cpu().numpy()),
                phase+'_accuracy_domain_step':accuracy_score(labels.detach().cpu().numpy(),preds_class.detach().cpu().numpy()),
                phase+'_f1_classification_step':f1_score(labels.detach().cpu().numpy(), preds_class.detach().cpu().numpy()), 
                phase+'_f1_domain_step':f1_score(labels.detach().cpu().numpy(), preds_class.detach().cpu().numpy()),
                phase+'_auc_classification_step': roc_auc_score(labels.detach().cpu().numpy(), preds_class.detach().cpu().numpy()), 
                phase+'_auc_domain_step': roc_auc_score(labels.detach().cpu().numpy(), preds_class.detach().cpu().numpy())
            })
    return model, running_loss_class, all_y, all_preds, all_scores, running_loss_domain, all_domains, all_domain_preds, all_domain_scores # may god forgive me for this return statement

def calculate_epoch_metrics(loss, y, preds, scores):
    assert y.size(0) == preds.size(0)
    assert preds.size(0) == scores.size(0)
    loss = loss / y.size(0)
    acc = accuracy_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
    f1 = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
    auc = roc_auc_score(y.detach().cpu().numpy(), scores.detach().cpu().numpy())
    return loss, acc, f1, auc
    
def train_model(model, dataloaders, criterion, optimizer, save_dir, num_epochs=25, is_inception=False, limit_batches=-1):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_class = 0.0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print("Starting ", phase, " phase")
            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            pbar.set_description(f"{phase.upper()} - Epoch {epoch+1} / {num_epochs}")
            model, running_loss_class, all_y, all_preds, all_scores, running_loss_domain, all_domains, all_domain_preds, all_domain_scores = do_phase(phase, model, pbar, criterion=criterion, optimizer=optimizer, limit_batches=limit_batches)

            epoch_loss_class, epoch_acc_class, epoch_f1_class, epoch_auc_class = calculate_epoch_metrics(running_loss_class, all_y, all_preds, all_scores)
            epoch_loss_domain, epoch_acc_domain, epoch_f1_domain, epoch_auc_domain = calculate_epoch_metrics(running_loss_domain, all_domains, all_domain_preds, all_domain_scores)
    
            wandb.log({phase+'_loss_classification_epoch':epoch_loss_class, phase+'_loss_domain_epoch':epoch_loss_domain, 
                       phase+'_accuracy_classification_epoch':epoch_acc_class, phase+'_accuracy_domain_epoch':epoch_acc_domain,
                       phase+'_f1_classification_epoch':epoch_f1_class, phase+'_f1_domain_epoch':epoch_f1_domain,
                       phase+'_auc_classification_epoch': epoch_auc_class, phase+'_auc_domain_epoch':epoch_auc_domain})    
            
            print('{} C-Loss: {:.4f} C-Acc: {:.4f} C-F1: {:.4f} C-AUC: {:.4f}'.format(phase, epoch_loss_class, epoch_acc_class, epoch_f1_class, epoch_auc_class))
            print('{} D-Loss: {:.4f} D-Acc: {:.4f} D-F1: {:.4f} D-AUC: {:.4f}'.format(phase, epoch_loss_domain, epoch_acc_domain, epoch_f1_domain, epoch_auc_domain))
            
            
            
            # deep copy the model
            if phase == 'val' and epoch_acc_class > best_acc_class:
                best_acc_class = epoch_acc_class
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir, "best_val_acc.pth")
                print("Val. acc. (class) improved; saving to", save_path)
                torch.save(model.state_dict(), save_path)
            latest_path = os.path.join(save_dir, "latest.pth")
            print("Saving most recent model to", latest_path)
            torch.save(model.state_dict(), latest_path)
            if phase == 'val':
                val_acc_history.append(epoch_acc_class)
            # TODO: SAVE THIS SOMEHOW

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc Class: {:4f}'.format(best_acc_class))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def build_optimizer(opt_name, model, **kwargs):
    return getattr(torch.optim, opt_name)(model.parameters(), **kwargs)

# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False

# def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#     model_ft = None
#     input_size = 0

#     if model_name == "resnet":
#         """ Resnet18
#         """
#         model_ft = models.resnet18(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "alexnet":
#         """ Alexnet
#         """
#         model_ft = models.alexnet(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "vgg":
#         """ VGG11_bn
#         """
#         model_ft = models.vgg11_bn(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "squeezenet":
#         """ Squeezenet
#         """
#         model_ft = models.squeezenet1_0(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         model_ft.num_classes = num_classes
#         input_size = 224

#     elif model_name == "densenet":
#         """ Densenet
#         """
#         model_ft = models.densenet121(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier.in_features
#         model_ft.classifier = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "inception":
#         """ Inception v3
#         Be careful, expects (299,299) sized images and has auxiliary output
#         """
#         model_ft = models.inception_v3(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         # Handle the auxilary net
#         num_ftrs = model_ft.AuxLogits.fc.in_features
#         model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
#         # Handle the primary net
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 299

#     else:
#         print("Invalid model name, exiting...")
#         exit()


#     model_ft_2head = TwoHeadResNet(model_ft) # TODO: generalize to multiple model types
#     return model_ft_2head, input_size


def get_dataloaders(dataset_name, root_dir, corr, seed, batch_size, num_workers, test_only=False):
    if test_only:
        train_dl = None
    if dataset_name == 'mnist':
        if not test_only:
            train_dataset = datasets.CorrelatedMNIST(
                    mode="train",
                    spurious_match_prob=corr,
                    seed=seed,
                    root_dir=root_dir,
                )
            train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_dataset = datasets.CorrelatedMNIST(
                mode="test",
                spurious_match_prob=corr,
                seed=seed,
                root_dir=root_dir,
            )
        test_dl = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name in ['camelyon17', 'iwildcam']:
        if not test_only:
            if dataset_name == 'camelyon17':
                train_dataset = datasets.CorrelatedCamelyon17(
                    mode="train",
                    transform=None, # matybe change
                    root_dir=root_dir, # make configurable
                    domains=[0, 3],
                    normalize=True,
                    seed=42,
                )
            elif dataset_name == 'iwildcam':
                train_dataset = datasets.CorrelatedIWildcam(
                    mode="train",
                    transform=None, # matybe change
                    root_dir=root_dir, # make configurable
                    domains=[139, 230],
                    normalize=True,
                    seed=42,
                )

            train_dl = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=train_dataset.get_correlation_sampler(args.corr),
            )

        if dataset_name == 'camelyon17':
            test_dataset = datasets.CorrelatedCamelyon17(
                    mode="id_val",
                    transform=None, # matybe change
                    root_dir=root_dir, # make configurable
                    domains=[0, 3],
                    normalize=True,
                    seed=42,
                )
        elif dataset_name == 'iwildcam':
            test_dataset = datasets.CorrelatedIWildcam(
                    mode="id_val",
                    transform=None, # matybe change
                    root_dir=root_dir, # make configurable
                    domains=[139, 230],
                    normalize=True,
                    seed=42,
                )
        test_dl = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=test_dataset.get_correlation_sampler(corr),
                )
    else:
        raise ValueError(f"Dataset with name '{dataset_name}' not supported.")
    return train_dl, test_dl


def run_experiment(
        dataset_name,
        corr,
        model_name,  # [resnet, alexnet, vgg, squeezenet, densenet, inception]
        n_epochs,
        batch_size,
        opt_name,
        lr,
        root_dir,
        save_dir,
        opt_kwargs,
        limit_batches=-1,
        seed=42,
        num_workers=4,
):
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    
    
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = root_dir#"./data/"

    # Number of classes in the dataset
    num_classes = 2

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    #----------------------------------- LOAD DATA -----------------------------------
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # # Create training and validation dataloaders
    # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    train_dl, val_dl = get_dataloaders(dataset_name, root_dir,corr, seed, batch_size, num_workers)

    dataloaders_dict = {
        "train": train_dl,
        "val": val_dl,
    } # b-b-but wait! why are we evaluating on test straight-up? bc we're not doing model selection -- idc how good a model does!


    #----------------------------------- OPTIMIZER -----------------------------------
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = build_optimizer(opt_name, model_ft, lr=lr, **opt_kwargs)

    # THESE TWO ARE THE RIGHT SETTINGS FOR IWILDCAM AND CAMELYON -- CAN'T REMEMBER WHICH IS WHICH
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # Setup the loss fxn
    criterion = nn.BCELoss()

    # Train and evaluate
    os.makedirs(save_dir, exist_ok=True)
    model_ft, hist = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer_ft,
            save_dir,
            num_epochs=n_epochs,
            is_inception=(model_name == "inception"),
            limit_batches=limit_batches,
        )


if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--dataset", type=str, choices=['mnist','camelyon17', 'iwildcam'], default='mnist')
    psr.add_argument("--corr", type=float, default=0.7)
    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--wandb_expt_name", type=str, required=True)

    psr.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    psr.add_argument("--model_name", choices=['resnet','densenet'], type=str, required=True)
    psr.add_argument("--batch_size", default=32, type=int)
    psr.add_argument("--n_epochs", default=5, type=int)
    psr.add_argument("--opt_name", default="SGD", type=str)
    psr.add_argument("--lr", type=float, required=True)
    psr.add_argument("--opt_kwargs", type=str, nargs='+', default={})
    psr.add_argument("--limit_batches", type=int, default=-1)
    psr.add_argument("--root_dir", type=str, default='/scratch/eecs542f21_class_root/eecs542f21_class/shared_data/dssr_datasets/WildsData')

    psr.add_argument("--save_dir", type=str, default='/scratch/eecs542f21_class_root/eecs542f21_class/shared_data/dssr_datasets/saved_models/')

    args = psr.parse_args()
    parsed_opt_kwargs = parse_argdict(args.opt_kwargs)
    
    metadata = {**vars(args), **parsed_opt_kwargs}
    
    wandb.init(project="eecs542", entity="eecs542", config=metadata)
    wandb.run.name = args.wandb_expt_name

    run_experiment(
        args.dataset,
        args.corr,
        args.model_name,
        args.n_epochs,
        args.batch_size,
        args.opt_name,
        args.lr,
        args.root_dir,
        os.path.join(args.save_dir, args.dataset, args.wandb_expt_name), # <save_dir>/<dataset>/<run>
        parsed_opt_kwargs,
        limit_batches=args.limit_batches,
        seed=args.seed,
        num_workers=args.num_workers,
    )

