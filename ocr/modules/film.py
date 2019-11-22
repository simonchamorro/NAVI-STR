from torch import nn


class FiLMGen(nn.Module):
    def __init__(self,
        input_dim=200,
        num_layers=1,
        cond_feat_size=18944, # (4 num conditioning layers by two * 128 dimension) + (8 * 256) + (20 * 512) + (12 * 512)
        emb_dim=1000,
        transformer=False):
        super(FiLMGen, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.cond_feat_size = cond_feat_size  # gammas and betas
        self.num_layers = num_layers
        if transformer:
            self.layers = nn.TransformerEncoderLayer(self.input_dim, 10, dim_feedforward=128)
        else:
            self.layers = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.cond_feat_size)
            )


    def forward(self, x):
        import pdb; pdb.set_trace()
        x = self.layers(x)
        return x
