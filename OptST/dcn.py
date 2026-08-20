import torch
import torch.nn as nn
from layers import CrossNetwork, MultiLayerPerceptron


class DCNModel(nn.Module):
    def __init__(self, feature_dims: dict, embed_dim: int = 8, sid_embed_dim: int = 8,
                 cross_layers: int = 3, deep_layers: list = [16, 16],
                 dropout: float = 0.2):
        super(DCNModel, self).__init__()

        self.feature_dims = feature_dims
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(feature_dims['user_id'], embed_dim)
        self.item_embedding = nn.Embedding(feature_dims['item_id'], embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight.data)
        nn.init.xavier_uniform_(self.item_embedding.weight.data)

        self.dnn_dim = 2 * embed_dim + sid_embed_dim

        self.cross_network = CrossNetwork(self.dnn_dim, cross_layers)

        self.deep_network = MultiLayerPerceptron(
            input_dim=self.dnn_dim,
            embed_dims=deep_layers,
            dropout=dropout,
            use_bn=True,
            output_layer=False
        )

        final_dim = self.dnn_dim + deep_layers[-1]
        self.combination = nn.Linear(final_dim, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, features: dict, sid_feature: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(features['user_id'])   # (batch, embed_dim)
        i_emb = self.item_embedding(features['item_id'])   # (batch, embed_dim)

        concat_features = torch.cat([u_emb, i_emb, sid_feature], dim=1)  # (batch, dnn_dim)

        cross_out = self.cross_network(concat_features)    # (batch, dnn_dim)

        deep_out = self.deep_network(concat_features)      # (batch, deep_layers[-1])

        combined = torch.cat([cross_out, deep_out], dim=1)  # (batch, dnn_dim + deep_layers[-1])
        logit = self.combination(combined)                   # (batch, 1)

        return self.sigmoid(logit)
