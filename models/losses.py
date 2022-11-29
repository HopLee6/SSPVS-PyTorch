import torch.nn as nn


class mse(nn.Module):
    def __init__(self, args):
        super(mse, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs):
        frame_score = inputs["frame_score"]  # b,l,1
        gtscore = inputs["gtscore"].unsqueeze(2)  # b,l,1
        gtscore -= gtscore.min(dim=1, keepdim=True)[0]
        gtscore /= gtscore.max(dim=1, keepdim=True)[0]
        loss = self.loss(frame_score, gtscore)
        return loss


def loss(args):
    classes = globals()
    if not args.loss in classes:
        raise NotImplementedError("not implemented loss function " + args.loss)
    loss_cls = classes[args.loss]
    return loss_cls(args)
