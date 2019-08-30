from torch import nn


class FiLMGen(nn.Module):
    def __init__(self,
        input_dim=200,
        cond_feat_size=18944, # (4 num conditioning layers by two * 128 dimension) + (8 * 256) + (20 * 512) + (12 * 512)
        emb_dim=1000,
        init_xavier=False):
        super(FiLMGen, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.cond_feat_size = cond_feat_size  # gammas and betas
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.cond_feat_size)
        )
        if init_xavier:
            self.layers.apply(self.init_weights)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.layers(x)
        return x
