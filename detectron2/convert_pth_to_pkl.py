from detectron2.modeling import build_model
import torch
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
import pickle

def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file('configs/PascalVOC-Detection/faster_rcnn_X_101_FPN.yaml')
    cfg.freeze()
    return cfg

args = default_argument_parser().parse_args()
cfg = setup()

model = build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
torch.save(model.state_dict(), 'X101-FPN-pretrained-wsi.pkl')

