import torch


class NeRFSmall(torch.nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = torch.nn.ModuleList(sigma_net)

        self.color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = 1
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 1
            else:
                out_dim = hidden_dim_color

            self.color_net.append(torch.nn.Linear(in_dim, out_dim, bias=True))

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, inplace=True)

        sigma = h
        return sigma
