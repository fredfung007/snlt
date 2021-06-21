# COPYRIGHT 2021. Fred Fung. Boston University.
import torch
from transformers import BertTokenizer, BertModel

from region_proposal_network.depthwise_rpn import DepthwiseRPN


class MultiNLRPN(torch.nn.Module):
    def __init__(self, lang_cfg, anchor_num, in_channels, weighted=True):
        super(MultiNLRPN, self).__init__()
        if lang_cfg.MODEL != 'bert':
            raise AssertionError('MULTI LANG RPN FOR BERT LANGUAGE MODELS ONLY. CONFIG.LANG.MODEL SHOULD BE "bert".')
        self.bert_tokenizer = BertTokenizer.from_pretrained(lang_cfg.BERT_CHECKPOINT)
        self.bert_model = BertModel.from_pretrained(lang_cfg.BERT_CHECKPOINT)
        self.l2_rpn = DepthwiseRPN(anchor_num, in_channels[0], in_channels[0])
        self.l3_rpn = DepthwiseRPN(anchor_num, in_channels[1], in_channels[1])
        self.l4_rpn = DepthwiseRPN(anchor_num, in_channels[2], in_channels[2])
        self.weighted = weighted

        if self.weighted:
            self.cls_weight = torch.nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)
            self.loc_weight = torch.nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)
            self.nl_cls_weight = torch.nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)
            self.nl_loc_weight = torch.nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)

    def forward(self, zf_s, x_fs, phrase):
        tokens = self.bert_tokenizer.batch_encode_plus(phrase, padding='longest', return_tensors='pt')
        embeds = self.bert_model(tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda())[0]
        embeds = torch.mean(embeds, dim=1)
        embeds = embeds.view([-1, 768, 1, 1])
        l2_output = self.l2_rpn(zf_s[0], x_fs[0], embeds[:, :256, :, :])
        l3_output = self.l3_rpn(zf_s[1], x_fs[1], embeds[:, 256:512, :, :])
        l4_output = self.l4_rpn(zf_s[2], x_fs[2], embeds[:, 512:768, :, :])
        cls = [l2_output['cls'], l3_output['cls'], l4_output['cls']]
        reg = [l2_output['reg'], l3_output['reg'], l4_output['reg']]
        nl_cls = [l2_output['nl_cls'], l3_output['nl_cls'], l4_output['nl_cls']]
        nl_reg = [l2_output['nl_reg'], l3_output['nl_reg'], l4_output['nl_reg']]
        if self.weighted:
            cls_weight = torch.nn.functional.softmax(self.cls_weight, 0)
            loc_weight = torch.nn.functional.softmax(self.loc_weight, 0)
            nl_cls_weight = torch.nn.functional.softmax(self.nl_cls_weight, 0)
            nl_loc_weight = torch.nn.functional.softmax(self.nl_loc_weight, 0)
            cls = self._weighted_avg(cls, cls_weight)
            reg = self._weighted_avg(reg, loc_weight)
            nl_cls = self._weighted_avg(nl_cls, nl_cls_weight)
            nl_reg = self._weighted_avg(nl_reg, nl_loc_weight)
        else:
            cls = self._avg(cls)
            reg = self._avg(reg)
            nl_cls = self._avg(nl_cls)
            nl_reg = self._avg(nl_reg)
        return {'cls': cls, 'reg': reg, 'nl_cls': nl_cls, 'nl_reg': nl_reg}

    @staticmethod
    def _avg(lst):
        return sum(lst) / len(lst)

    @staticmethod
    def _weighted_avg(lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s
