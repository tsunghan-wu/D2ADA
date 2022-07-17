from torch import nn
from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV2", "SimpleAuxHead"]


class DeepLabV2(_SimpleSegmentationModel):
    """
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
    """
    pass


class DeepLabHeadV2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHeadV2, self).__init__()
        self.aspp = ASPP(in_channels, 256, [6, 12, 18, 24])
        self.final = nn.Conv2d(256, num_classes, 1)
        self.return_feat = False

    def forward(self, feature):
        point_feature = self.aspp(feature)
        output = self.final(point_feature)
        if self.return_feat:
            return point_feature, output
        else:
            return output


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPConv(in_channels, out_channels, rate4))
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        x = x['out']
        res = []
        for conv in self.convs:
            res.append(conv(x))
        return sum(res)


class SimpleAuxHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(SimpleAuxHead, self).__init__()
        self.classifier = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        return self.classifier(x)
