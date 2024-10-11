import onnxruntime as ort
import numpy as np
import platform
from pathlib import Path
from pathlib import PureWindowsPath
from typing import List, Tuple

class ONNXInference:
    def __init__(self, name: str):
        self.name: str = name

        models_dir_path: str = str(Path(__file__).parent.parent / 'models')
        print(models_dir_path)
        path: str = f'{models_dir_path}/{name}'
        if platform.system() == 'Windows':
            model_path: str = str(PureWindowsPath(path))
        elif platform.system() == 'Linux':
            model_path: str = path

        self.session: ort.InferenceSession = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.model_inputs: List[ort.NodeArg] = self.session.get_inputs()
        self.input_type: np.dtype = np.float32
        self.max_batch_size: int = 1
        
    def __call__(self, inputs: np.ndarray) -> List[np.ndarray]:
        data: np.ndarray = np.array(inputs).astype(self.input_type)
        if data.shape[0] <= self.max_batch_size:  # batch <= max batch size
            results: List[np.ndarray] = self.session.run(None, {self.model_inputs[0].name: data})
        return results

    def get_inputs(self) -> Tuple[int, int, int]:
        input_shape: Tuple[int, int, int, int] = self.model_inputs[0].shape
        return input_shape[1], input_shape[2], input_shape[3]