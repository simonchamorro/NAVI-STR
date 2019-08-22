from torch import nn


class FiLMGen(nn.Module):
    def __init__(self,
        input_dim=200,
        cond_feat_size=18944, # (4 num conditioning layers by two * 128 dimension) + (8 * 256) + (20 * 512) + (12 * 512)
        emb_dim=1000):
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


    def forward(self, x):
        x = self.layers(x)
        return x

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    import pdb; pdb.set_trace()
    x = (gammas.view(x.shape[0], x.shape[1], 1, 1) * x) + betas.view(x.shape[0], x.shape[1], 1, 1)
    return x
