# COPYRIGHT 2021. Fred Fung. Boston University.
import torch

from region_proposal_network.adjustment_layer import AdjustAllLayer
from region_proposal_network.loss import weight_l1_loss, select_cross_entropy_loss, triplet_loss
from region_proposal_network.multi_nl_rpn import MultiNLRPN
from third_party.resnet_atrous import ResNet50
from tracker.siamese_tracker import SiameseTracker
from utils.tracking_inference import log_softmax


class SiamRPNPPMultiNLRPN(SiameseTracker):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.backbone = ResNet50(used_layers=[2, 3, 4])
        self.adjustment_layer = AdjustAllLayer(in_channels=[512, 1024, 2048], out_channels=[256, 256, 256])
        self.nl_multi_rpn = MultiNLRPN(self.cfg.LANG, self.cfg.ANCHOR.ANCHOR_NUM, in_channels=[256, 256, 256])

    def forward(self, data):
        template = data["template"].cuda()
        search = data["search"].cuda()
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.adjustment_layer(zf)
        xf = self.adjustment_layer(xf)
        rpn_output = self.nl_multi_rpn(zf, xf, data["phrase"])
        cls = log_softmax(rpn_output["cls"])
        nl_cls = log_softmax(rpn_output["nl_cls"])
        return {"cls": cls, "reg": rpn_output["reg"], "nl_cls": nl_cls, "nl_reg": rpn_output["nl_reg"]}

    def get_trainable_weights(self, train_backbone, finetune_bert=False):
        trainable_params = []
        if train_backbone:
            trainable_params += [{"params": filter(lambda x: x.requires_grad, self.backbone.parameters()),
                                  "lr": self.cfg.TRAIN.LR.BACKBONE_LAYERS_LR}]
            trainable_params += [{"params": self.adjustment_layer.parameters()}]
        trainable_params += [{"params": self.nl_multi_rpn.parameters()}]
        if finetune_bert:
            raise NotImplementedError("Finetuning BERT is not implemented.")
        return trainable_params

    def compute_loss(self, data):
        rpn_output = self.forward(data)
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        phrase_present = torch.tensor([0 if phrase == '' else 1 for phrase in data['phrase']],
                                      requires_grad=False).cuda()
        loc_loss = weight_l1_loss(rpn_output['reg'], label_loc, label_loc_weight)
        if self.cfg.TRAIN.USE_TRIPLET_LOSS:
            with torch.no_grad():
                cls_loss = select_cross_entropy_loss(rpn_output['cls'].detach(), label_cls)
            t_loss = triplet_loss(rpn_output['cls'], label_cls, margin=0.4)
            total_loss = self.cfg.TRAIN.LR.CLS_WEIGHT * t_loss + self.cfg.TRAIN.LR.LOC_WEIGHT * loc_loss
        else:
            cls_loss = select_cross_entropy_loss(rpn_output['cls'], label_cls)
            total_loss = self.cfg.TRAIN.LR.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LR.LOC_WEIGHT * loc_loss

        nl_loc_loss = weight_l1_loss(rpn_output['nl_reg'], label_loc, label_loc_weight, phrase_present)
        if self.cfg.TRAIN.USE_TRIPLET_LOSS:
            with torch.no_grad():
                nl_cls_loss = select_cross_entropy_loss(rpn_output['nl_cls'].detach(), label_cls, phrase_present)
            nl_t_loss = triplet_loss(rpn_output['nl_cls'], label_cls, margin=0.4)
            nl_total_loss = self.cfg.TRAIN.LR.CLS_WEIGHT * nl_t_loss + self.cfg.TRAIN.LR.LOC_WEIGHT * nl_loc_loss
        else:
            nl_cls_loss = select_cross_entropy_loss(rpn_output['nl_cls'], label_cls, phrase_present)
            nl_total_loss = self.cfg.TRAIN.LR.CLS_WEIGHT * nl_cls_loss + self.cfg.TRAIN.LR.LOC_WEIGHT * nl_loc_loss
        losses = {'nl_total_loss': nl_total_loss, 'nl_cls_loss': nl_cls_loss, 'nl_loc_loss': nl_loc_loss,
                  'vis_total_loss': total_loss, 'vis_cls_loss': cls_loss, 'vis_loc_loss': loc_loss,
                  'total_loss': total_loss + nl_total_loss}
        if self.cfg.TRAIN.USE_TRIPLET_LOSS:
            losses['vis_t_loss'] = t_loss
            losses['nl_t_loss'] = nl_t_loss
        return losses
