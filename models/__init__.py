import models.segmentation as network
import torch.nn as nn


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if hasattr(m, 'weight'):
                m.weight.requires_grad_(False)
            if hasattr(m, 'bias'):
                m.bias.requires_grad_(False)
            m.eval()


def get_model(model, num_classes, output_stride, separable_conv):
    assert model in ['deeplabv3_resnet50', 'deeplabv3plus_resnet50', 'deeplabv3_resnet101',
                     'deeplabv3plus_resnet101', 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                     'deeplabv2_resnet101', 'deeplabv2_mobilenet']

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv2_resnet101': network.deeplabv2_resnet101,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv2_mobilenet': network.deeplabv2_mobilenet,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
    }

    # Set up model
    net = model_map[model](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in model:
        network.convert_to_separable_conv(net.classifier)
    set_bn_momentum(net.backbone, momentum=0.1)

    return net
