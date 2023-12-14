from model.SMDR_IS import SMDR_IS

import pytorch_lightning as pl
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import torch
from loss.Perceptual import PerceptualLoss


from pytorch_lightning import seed_everything

# Set seed
seed = 42  # Global seed set to 42
seed_everything(seed)
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('all',name='all9')


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss



class model(pl.LightningModule):

    def __init__(self):
        super(model, self).__init__()

        # loss_function
        self.loss_L1 = L1_Charbonnier_loss().cuda()
        self.loss_Pe = PerceptualLoss().cuda()
        self.loss_mse = torch.nn.MSELoss(reduction='mean').cuda()
        self.model = SMDR_IS()

    def forward(self, x):
        y = self.model(x)
        y1 = y[0]
        y1 = y1[:, :, 0:x.shape[2], 0:x.shape[3]]
        return y1
