
import torch.nn as nn

class ZipLoss(nn.Module):
    
    def __init__(
        self,
        loss_functions: list,
    ) -> None:
        super().__init__()
        self.losses = nn.ModuleList(loss_functions)

    def forward(
        self,
        pred: list,
        target: list,
        squash: bool = True,
    ) -> tuple:
        out = ()
        for i, loss in enumerate(self.losses):
            out += (loss(pred[i], target[i], squash),)

        return out

        

    


