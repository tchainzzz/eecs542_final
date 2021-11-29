import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


#called resnet but also works for other architectures
class TwoHeadResNet(torch.nn.Module):
    def __init__(self, resnetModel):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoHeadResNet, self).__init__()
        self.l_input_size = resnetModel.fc.in_features
        self.resnetBackbone = torch.nn.Sequential(*(list(resnetModel.children())[:-1]))

        self.classHead = torch.nn.Linear(self.l_input_size, 1)
        self.domainHead = torch.nn.Linear(self.l_input_size, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        backboneOut = self.resnetBackbone(x)
        backboneOut = backboneOut.view(-1, self.l_input_size)
        classOut = self.classHead(backboneOut)
        domainOut = self.domainHead(backboneOut)
        return torch.sigmoid(classOut), torch.sigmoid(domainOut)


class TwoHeadDenseNet(torch.nn.Module):
    def __init__(self, densenetModel):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoHeadDenseNet, self).__init__()
        self.l_input_size = densenetModel.classifier.in_features
        self.densenetBackbone = torch.nn.Sequential(*(list(densenetModel.children())[:-1]))


        self.classHead = torch.nn.Linear(self.l_input_size, 1)
        self.domainHead = torch.nn.Linear(self.l_input_size, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        print("Shape of training input:")
        print(x.shape)
        # only works on pictures at least 32 in size, mnist is 28
        if x.shape[2] < 32 or x.shape[3] < 32:
             x = F.pad(x, (2,2,2,2), "constant", 0)

        backboneOut = self.densenetBackbone(x)
        backboneOut = backboneOut.view(-1, self.l_input_size)
        classOut = self.classHead(backboneOut)
        domainOut = self.domainHead(backboneOut)
        return torch.sigmoid(classOut), torch.sigmoid(domainOut)

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

        model_ft_2head = TwoHeadResNet(model_ft) # TODO: generalize to multiple model types

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

        model_ft_2head = TwoHeadDenseNet(model_ft) # TODO: generalize to multiple model types

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


    
    return model_ft_2head, input_size
