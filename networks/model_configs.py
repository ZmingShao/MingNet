import ml_collections


def get_vit_config():
    config = ml_collections.ConfigDict()
    config.channels = {'in': 1024, 'out': 1024}
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    return config
