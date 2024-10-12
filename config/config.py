import yaml
import os

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Union

@dataclass
class Config:
    classes: List[int] = field(default_factory=list)
    fps: int = 30
    height: int = 1080
    width: int = 1920
    source: Union[int, str] = 0  # Can be int for USB camera or str for IP camera URL
    model_name: str = "yolo11s.onnx"
    conf_threshold: float = 0.25
    nms_threshold: float = 0.75
    colors: List[Tuple[int, int, int]] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    output_directory: str = "output"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'Config':
        with open(yaml_file, 'r') as f:
            config_data: Dict = yaml.safe_load(f)
        return cls(**config_data)

config: Config = Config.from_yaml('config/config.yaml')
