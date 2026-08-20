import torch
import torch.nn as nn


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)              # (batch, 1)
            x = x0 * xw + self.b[i] + x    # Broadcasting + residual.
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0., use_bn=False, output_layer=True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
