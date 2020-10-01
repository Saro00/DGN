import torch
import torch.nn as nn

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout, device=device)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation='relu', last_activation='none',
                 dropout=0., mid_b_norm=False, last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm,
                                                device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    b_norm=mid_b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'
