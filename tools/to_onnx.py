import argparse
import os
import pprint
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/rugd/finetune_ddrnet23.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


class onnx_net(nn.Module):
    def __init__(self, model):
        super(onnx_net, self).__init__()
        self.backone = model

    def forward(self, x):
        x1, x2 = self.backone(x)
        y = F.interpolate(x1, size=(180, 320), mode='bilinear')
        # y = F.softmax(y, dim=1)
        y = torch.argmax(y, dim=1)

        return y


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    net = onnx_net(model)
    net = net.eval()

    # x = torch.randn((1, 3, 512, 384))
    x = torch.randn((1, 3, 180, 320))
    torch_out = net(x)

    output_path = "output/ddrnet23.onnx"
    torch.onnx.export(net,
                      x,
                      output_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['inputx'],
                      output_names=['outputy'],
                      verbose=True,
                      )


if __name__ == '__main__':
    main()
