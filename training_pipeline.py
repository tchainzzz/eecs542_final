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
from torchvision import models, transforms
from utils import parse_argdict

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, limit_batches=-1):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            all_y = torch.Tensor()
            all_scores = torch.Tensor()
            all_preds = torch.Tensor()

            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            pbar.set_description(f"{phase.upper()} - Epoch {epoch+1} / {num_epochs}")
            for i, (inputs, labels, domains) in pbar:
                if i == limit_batches: break
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scores, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                all_y = torch.cat((all_y, labels))
                all_preds = torch.cat((all_preds, preds))
                all_scores = torch.cat((all_scores, scores))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({
                    "loss": running_loss / all_y.size(0),
                    "acc": accuracy_score(all_y.detach().numpy(), all_preds.detach().numpy()),
                    "f1": f1_score(all_y.detach().numpy(), all_preds.detach().numpy()),
                    "auc": roc_auc_score(all_y.detach().numpy(), all_scores.detach().numpy())
                })
                # TODO: CALCULATE METRICS FOR AUX_OUTPUTS AS WELL

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_y.detach().numpy(), all_preds.detach().numpy())
            epoch_f1 = f1_score(all_y.detach().numpy(), all_preds.detach().numpy())
            epoch_auc = roc_auc_score(all_y.detach().numpy(), all_scores.detach().numpy())
            # TODO: SAVE THESE

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def build_optimizer(opt_name, model, **kwargs):
    return getattr(torch.optim, opt_name)(model.parameters(), **kwargs)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_eval_model(
        dataloader,
        model_name,  # [resnet, alexnet, vgg, squeezenet, densenet, inception]
        n_epochs,
        opt_name,
        lr,
        opt_kwargs,
        limit_batches=-1,
):
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)


    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./data/"

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

    dataloaders_dict = {x: dataloader for x in ['train', 'val']}


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
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer_ft,
            num_epochs=n_epochs,
            is_inception=(model_name == "inception"),
            limit_batches=limit_batches,
        )


if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--dataset", type=str, choices=['mnist'], default='mnist')
    psr.add_argument("--corr", type=float, default=0.7)
    psr.add_argument("--seed", type=int, default=42)

    psr.add_argument("--num-workers", default=os.cpu_count(), type=int)
    psr.add_argument("--model-name", type=str, required=True)
    psr.add_argument("--batch-size", default=16, type=int)
    psr.add_argument("--n-epochs", default=50, type=int)
    psr.add_argument("--opt-name", default="SGD", type=str)
    psr.add_argument("--lr", type=float, required=True)
    psr.add_argument("--opt-kwargs", type=str, nargs='+', default={})
    psr.add_argument("--limit-batches", type=int, default=-1)

    args = psr.parse_args()

    if args.dataset == 'mnist':

        dataset = datasets.CorrelatedMNIST(
                spurious_match_prob=args.corr,
                seed=args.seed,
            )
        dataloader = DataLoader(dataset, batch_size=args.batch_size) # TODO: need to separately define train + eval for each expeirment

        train_eval_model(
            dataloader,
            args.model_name,
            args.n_epochs,
            args.opt_name,
            args.lr,
            parse_argdict(args.opt_kwargs),
            args.limit_batches,
        )
