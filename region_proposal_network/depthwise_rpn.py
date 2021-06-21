# COPYRIGHT 2021. Fred Fung. Boston University.
import torch


class DepthwiseRPN(torch.nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, hidden=256):
        super(DepthwiseRPN, self).__init__()
        self.cls_conv_kernel = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.cls_conv_search = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 2 * anchor_num, kernel_size=1)
        )
        self.reg_conv_kernel = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.reg_conv_search = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 4 * anchor_num, kernel_size=1)
        )

        self.nl_cls_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden, hidden, kernel_size=5, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 2 * anchor_num, kernel_size=1)
        )
        self.nl_reg_head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden, hidden, kernel_size=5, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 4 * anchor_num, kernel_size=1)
        )

    def forward(self, z_f, x_f, nl_kernel):
        kernel = self.cls_conv_kernel(z_f)
        search = self.cls_conv_search(x_f)
        feature = xcorr_depthwise(search, kernel)
        cls = self.cls_head(feature)
        feature = xcorr_depthwise(search, nl_kernel)
        nl_cls = self.nl_cls_head(feature)
        kernel = self.reg_conv_kernel(z_f)
        search = self.reg_conv_search(x_f)
        feature = xcorr_depthwise(search, kernel)
        reg = self.reg_head(feature)
        feature = xcorr_depthwise(search, nl_kernel)
        nl_reg = self.nl_reg_head(feature)
        return {'cls': cls, 'reg': reg, 'nl_cls': nl_cls, 'nl_reg': nl_reg}


def xcorr_depthwise(search, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    search = search.view(1, batch * channel, search.size(2), search.size(3))
    kernel = kernel.reshape(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = torch.nn.functional.conv2d(search, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
