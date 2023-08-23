import os

import paddle
from paddle import inference

# PlaceType, PrecisionType, convert_to_mixed_precision


inference.convert_to_mixed_precision(
    os.path.join("./resnet50", "inference.pdmodel"),
    os.path.join("./resnet50", "inference.pdiparams"),
    os.path.join("./resnet50", "mixed_inference.pdmodel"),
    os.path.join("./resnet50", "mixed_inference.pdiparams"),
    inference.PrecisionType.Half,
    inference.PlaceType.GPU,
    False,
)
