# COPYRIGHT 2021. Fred Fung. Boston University.
import torch


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return torch.tensor(0., requires_grad=False)
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return torch.nn.functional.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label, phrase_present=None):
    if phrase_present is not None:
        label = label.transpose(0, -1)
        label = phrase_present * (1 + label) - 1
        label = label.transpose(0, -1)
        label = label.type(torch.int64)
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = torch.nonzero(label.data.eq(1), as_tuple=False).squeeze()
    neg = torch.nonzero(label.data.eq(0), as_tuple=False).squeeze()
    pos = pos.cuda()
    neg = neg.cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight, phrase_present=None):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    if phrase_present is not None:
        diff = diff.transpose(0, -1)
        diff = phrase_present * diff
        diff = diff.transpose(0, -1)
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    if phrase_present is not None:
        count = torch.sum(phrase_present)
        if not count.is_nonzero():
            count += 1
    else:
        count = b
    return loss.sum().div(count)


def triplet_loss(pred, label, phrase_present=None, margin=0.4):
    if phrase_present is not None:
        label = label.transpose(0, -1)
        label = phrase_present * (1 + label) - 1
        label = label.transpose(0, -1)
        label = label.type(torch.int64)
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1, 2)
    score = torch.nn.functional.softmax(pred, dim=-1)[:, :, 1]
    label = label.view(batch_size, -1)
    pos = label.eq(1).nonzero(as_tuple=False).squeeze()
    neg = label.eq(0).nonzero(as_tuple=False).squeeze()
    pos = pos.cuda()
    neg = neg.cuda()
    if len(pos.size()) == 0 or pos.size() == torch.Size([0]) or len(neg.size()) == 0 or neg.size() == torch.Size([0]):
        return torch.tensor(0., requires_grad=False)
    pos_pred = torch.stack([score[batch, index] for batch, index in pos])
    neg_pred = torch.stack([score[batch, index] for batch, index in neg])
    pos_length = pos.size()[0]
    neg_length = neg.size()[0]
    pos_pred = pos_pred.repeat(neg_length).view(neg_length, pos_length)
    neg_pred = neg_pred.repeat(pos_length).view(pos_length, neg_length).transpose(0, 1)
    distance = neg_pred - pos_pred + margin
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
    return loss
