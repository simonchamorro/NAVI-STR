from torch import nn


class FiLMGen(nn.Module):
    def __init__(self,
        input_dim=40 * 20 + 7 * 4,
        num_modules=4,
   		module_num_layers=11, # sum [1, 2, 5, 3]
    	module_dim=512):
        super(FiLMGen, self).__init__()
        self.num_models = num_modules
        self.module_num_layers = module_num_layers
        self.module_dim = module_dim
        self.cond_feat_size = 2 * 2 * 4 * 128 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.cond_feat_size)
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
    return (gammas.view(x.shape[0], x.shape[1], 1, 1) * x) + betas.view(x.shape[0], x.shape[1], 1, 1)
