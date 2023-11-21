import torch
import torch.nn as nn
import copy

class PatchEmbedding(nn.Module):
    """ -----Patch Embedding and position Embedding----- #
    # Apply patch embeding and position embeding on input hazy images.
    # Attributes:
    #       patch embedding: using Conv2d
    #       position embedding:a parameter with len = num_patch + 2(for mu and log_var)
    #       mu_tokens:token insert to the patch feature for mu
    #       log_var_tokens:token insert to the patch feature for log_var
    #       dropout:dropout for embeddings
    """
    def __init__(self, image_width=620,
                       image_height=460,
                       patch_size=20,
                       in_channels=3,
                       embed_dim=512,
                       dropout=0.):
        super().__init__()
        n_patches = (image_width // patch_size)*(image_height // patch_size)

        self.patch_embedding = nn.Conv2d(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.position_embeddings = nn.Parameter(data=torch.randn(1, n_patches+2, embed_dim),
                                               requires_grad=True)
        self.mu_tokens = nn.Parameter(data=torch.randn(1, 1, embed_dim),
                                      requires_grad=True)
        self.log_var_tokens = nn.Parameter(data=torch.randn(1, 1, embed_dim),
                                      requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.size()
        mu_tokens = self.mu_tokens.expand(B, -1, -1)
        log_var_tokens = self.log_var_tokens.expand(B, -1, -1)
        x = self.patch_embedding(x) # [B, embed_dim, H', W']
        x = torch.flatten(x, start_dim=2) # [B, embed_dim, n_patches]
        x = x.permute(0, 2, 1) # [B, n_patches, embed_dim]
        x = torch.cat((mu_tokens, log_var_tokens, x), 1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    """ Attention module
    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.
    Attributes:
        num_heads: number of heads
        attn_head_size: feature dim of single head
        all_head_size: feature dim of all heads
        qkv: a nn.Linear for q, k, v mapping
        scales: 1 / sqrt(single_head_feature_dim)
        out: projection of multi-head attention
        attn_dropout: dropout for attention
        proj_dropout: final dropout before output
        softmax: softmax op for attention
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        assert isinstance(embed_dim, int), (
            f"Expected the type of `embed_dim` to be {int}, but received {type(embed_dim)}.")
        assert isinstance(num_heads, int), (
            f"Expected the type of `num_heads` to be {int}, but received {type(num_heads)}.")

        assert embed_dim > 0, (
            f"Expected `embed_dim` to be greater than 0, but received {embed_dim}")
        assert num_heads > 0, (
            f"Expected `num_heads` to be greater than 0, but received {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if attn_head_size is not None:
            assert isinstance(attn_head_size, int), (
                f"Expected the type of `attn_head_size` to be {int}, "
                f"but received {type(attn_head_size)}.")
            assert attn_head_size > 0, f"Expected `attn_head_size` to be greater than 0," \
                                       f" but received {attn_head_size}."
            self.attn_head_size = attn_head_size
        else:
            self.attn_head_size = embed_dim // num_heads
            assert self.attn_head_size * num_heads == embed_dim, (
                f"`embed_dim` must be divisible by `num_heads`,"
                f" but received embed_dim={embed_dim}, num_heads={num_heads}.")

        self.all_head_size = self.attn_head_size * num_heads
        self.scales = self.attn_head_size ** -0.5

        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3,
                             bias=qkv_bias) # for q, k, v
        self.out = nn.Linear(self.all_head_size,
                             embed_dim)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_multihead(self, x):
        # x:[B, n_patches, all_head_size]
        B, N, _ = x.size()
        x = x.view(B, N, self.num_heads, self.attn_head_size)
        # x:[B, n_patches, num_heads, attn_head_size]
        x = x.permute(0, 2, 1, 3)
        # x:[B, num_heads, n_patches, attn_head_size]

        return x
    def forward(self, x):
        B, N, _ = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        # [B, N, all_head_size] * 3
        q, k, v = map(self.transpose_multihead, qkv)
        # q, k, v:[B, num_heads, n_patches, attn_head_size]

        attn = torch.matmul(q, k.permute(0, 1, 3, 2))
        # attn:[B, num_heads, n_patches, n_patches]
        attn = self.scales * attn
        attn = self.softmax(attn)
        # dropout
        attn = self.attn_dropout(attn) # attn: [B, num_heads, N]

        out = torch.matmul(attn, v)
        # out: [B, num_heads, n_patches, attn_head_size]
        out = out.permute(0, 2, 1, 3)
        # out: [B, n_patches, num_heads, attn_head_size]
        out = out.reshape(B, N, -1)
        out = self.out(out)

        return out


class Mlp(nn.Module):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim)

        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x


class EncoderLayer(nn.Module):
    """Encoder Layer
    Encoder layer contains attention, norm, mlp and residual
    Attributes:
        hidden_size: transformer feature dim
        attn_norm: nn.LayerNorm before attention
        mlp_norm: nn.LayerNorm before mlp
        mlp: mlp modual
        attn: attention modual
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=2.,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim,
                              num_heads,
                              attn_head_size,
                              qkv_bias,
                              dropout,
                              attention_dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim,
                       mlp_ratio,
                       dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x

class Encoder(nn.Module):
    """Transformer encoder
    Encoder encoder contains a list of EncoderLayer, and a LayerNorm.
    Attributes:
        layers: nn.LayerList contains multiple EncoderLayers
        encoder_norm: nn.LayerNorm which is applied after last encoder layer
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                         num_heads,
                                         attn_head_size=attn_head_size,
                                         qkv_bias=qkv_bias,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout)
            layer_list.append(copy.deepcopy(encoder_layer))

        self.layers = nn.ModuleList(layer_list)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)

        return out

class VisualTransformer(nn.Module):
    """ViT transformer
    ViT Transformer, classifier is a single Linear layer for finetune,
    For training from scratch, two layer mlp should be used.
    Classification is done using cls_token.
    Args:
        image_size: int, input image size, default: 224
        patch_size: int, patch size, default: 16
        in_channels: int, input image channels, default: 3
        num_classes: int, number of classes for classification, default: 1000
        embed_dim: int, embedding dimension (patch embed out dim), default: 768
        depth: int, number ot transformer blocks, default: 12
        num_heads: int, number of attention heads, default: 12
        mlp_ratio: float, ratio of mlp hidden dim to embed dim(mlp in dim), default: 4.0
        qkv_bias: bool, If True, enable qkv(nn.Linear) layer with bias, default: True
        dropout: float, dropout rate for linear layers, default: 0.
        attention_dropout: float, dropout rate for attention layers default: 0.
    """
    def __init__(self,
                 image_width=620,
                 image_height=460,
                 patch_size=20,
                 in_channels=3,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 attn_head_size=None,
                 mlp_ratio=2,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 num_dim=100
                 ):
        super().__init__()
        # create patch embedding with position embedding
        self.patch_embedding = PatchEmbedding(image_width,
                                              image_height,
                                              patch_size,
                                              in_channels,
                                              embed_dim,
                                              dropout)
        # create multi head self-attention layers
        self.encoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               attn_head_size,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout)
        self.mu = nn.Linear(embed_dim,
                            num_dim)
        self.log_var = nn.Linear(embed_dim,
                                 num_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        mu = self.mu(x[:, 0]) # take only mu_tokens
        log_var = self.log_var(x[:, 1])# take only log_var_tokens

        return mu, log_var
