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
        self.cond_feat_size = 2 * self.module_dim * self.module_num_layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, self.cond_feat_size)
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
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas
