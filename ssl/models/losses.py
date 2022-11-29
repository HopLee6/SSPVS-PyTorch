import torch
import torch.nn as nn


def acc(prob, label):
    return (prob.round() == label).float().mean()


def Hausdorff_dist(X, Y):
    len_X, len_Y = X.shape[0], Y.shape[0]
    X = torch.nn.functional.normalize(X, dim=2)
    Y = torch.nn.functional.normalize(Y, dim=2)
    X = X.unsqueeze(1).expand(-1, len_Y, -1, -1)
    Y = Y.unsqueeze(0).expand(len_X, -1, -1, -1)
    dist = torch.sqrt(torch.sum((X - Y) ** 2, dim=3))
    d_X_Y = dist.min(dim=1)[0].max(dim=0)[0]
    d_Y_X = dist.min(dim=0)[0].max(dim=0)[0]
    hd = torch.stack([d_X_Y, d_Y_X], dim=1).max(dim=1, keepdim=True)[0]
    return hd


class fuse(nn.Module):
    def __init__(self, args):
        super(fuse, self).__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.margin = args.margin

    def forward(self, inputs, mode="train"):
        predict_mask_feature, prob_v = inputs["predict_mask_feature"], inputs["prob_v"]
        encoded_frame, encoded_word = inputs["encoded_frame"], inputs["encoded_word"]
        hausdorff_dist = Hausdorff_dist(encoded_frame, encoded_word)
        label_mask_feature, label_v = inputs["masked_feature"], inputs["label"]
        label_v = label_v.float().unsqueeze(1)
        mask_predict_loss = self.mse(predict_mask_feature, label_mask_feature)
        loss_dict = {mode + "/mask_predict_loss": mask_predict_loss}
        contrast_loss = (
            label_v * hausdorff_dist**2
            + (1 - label_v) * ((self.margin - hausdorff_dist).relu()) ** 2
        ).mean()
        loss_dict[mode + "/contrast_loss"] = contrast_loss

        if mode == "train":
            bce_loss = self.bce(prob_v, label_v)
            loss = (
                bce_loss + self.alpha1 * mask_predict_loss + self.alpha2 * contrast_loss
            )

            loss_dict.update(
                {
                    mode + "/video_text_loss": bce_loss,
                    mode + "/loss": loss,
                }
            )
            return loss_dict
        video_text_acc = acc(prob_v, label_v)
        loss_dict[mode + "/video_text_acc"] = video_text_acc

        return loss_dict


def loss(args):
    classes = globals()
    if not args.loss in classes:
        raise NotImplementedError("not implemented loss function " + args.loss)
    loss_cls = classes[args.loss]
    return loss_cls(args)
