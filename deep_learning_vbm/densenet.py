import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



def _bn_function_factory(norm, relu, conv):
    """ 
    From https://github.com/Duplums/SMLvsDL. 
    Parameters : 
        norm : batch normalization function
        relu : relu activation function
        conv : convolution function
    Aim :
        creates a composite function (bn_function) that combines batch normalization, ReLU activation, 
        and convolution operations to process input tensors.
    Outputs : 
        nb_function : composite function that applies batch normalization (norm), ReLU activation (relu), 
                        and convolution (conv) operations to its input tensors.
    """
    def bn_function(*inputs):
        """
        Params :
            *inputs : a variable number of input tensors
        Aim :
            1. Concatenates the input tensors along the second dimension (dim=1). 
                This allows the batch normalization, ReLU, and convolution operations to be applied jointly across all input tensors.
            2. a. applies batch normalization (norm) to normalize the concatenated features. 
               b. applies ReLU activation (relu).
               c. applies convolution (conv) to the normalized and activated features to produce the final output.
        Outputs :
            output tensor obtained after applying the batch normalization, ReLU, and convolution operations.
        """
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    """ 
    This class is from https://github.com/Duplums/SMLvsDL. 
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        Aim : initializes the normalizations, activation functions and convolutional layers of the densenet.
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)

        if hasattr(self, 'concrete_dropout'):
            new_features = self.concrete_dropout(self.relu2(self.norm2(bottleneck_output)))
        else:
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

            if self.drop_rate > 0:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.Module):
    """ 
    From https://github.com/Duplums/SMLvsDL. 
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """ 
    This class is from https://github.com/Duplums/SMLvsDL. 
    """
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """ 
    Simplified from https://github.com/Duplums/SMLvsDL, where
    Densenet-BC model class, based on
    "Densely Connected Convolutional Networks" at https://arxiv.org/pdf/1608.06993.pdf
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1, in_channels=1):
        """
        Parameters:
            growth_rate : (int) how many filters to add each layer (`k` in paper)
            block_config : (list of 4 ints) how many layers in each pooling block
            num_init_features (int) the number of filters to learn in the first convolution layer
            bn_size : (int) multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
            drop_rate : (float) dropout rate after each dense layer
            num_classes : (int) number of classification classes
        """
        super(DenseNet, self).__init__()
        self.input_imgs = None
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
                )

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # apply transition unless it's the last block
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out.squeeze(dim=1)

    def get_current_visuals(self):
        return self.input_imgs

def densenet121(in_channels=1):
    """ 
    From https://github.com/Duplums/SMLvsDL, where the architecture of the 
    Densenet-121 model comes from the following paper :
    "Densely Connected Convolutional Networks" at https://arxiv.org/pdf/1608.06993.pdf
    """
    return DenseNet(in_channels=in_channels)



