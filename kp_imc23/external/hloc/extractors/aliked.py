
from ..utils.base_model import BaseModel
from .lightglue.aliked import ALIKED
class Aliked(BaseModel):
    default_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "nms_radius": 2,
    }

    def _init(self, conf):
        self.net = ALIKED(conf)

    def _forward(self, data):
        return self.net(data)
