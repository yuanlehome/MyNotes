import os

from paddle import inference

# inference.convert_to_mixed_precision(
#     os.path.join("./", "inference.pdmodel"),
#     os.path.join("./", "inference.pdiparams"),
#     os.path.join("./", "mixed_inference.pdmodel"),
#     os.path.join("./", "mixed_inference.pdiparams"),
#     inference.PrecisionType.Half,
#     inference.PlaceType.GPU,
#     False,
# )

inference.convert_to_mixed_precision(
    os.path.join("./", "mixed_inference.pdmodel"),
    os.path.join("./", "mixed_inference.pdiparams"),
    os.path.join("./", "white_mixed_inference.pdmodel"),
    os.path.join("./", "white_mixed_inference.pdiparams"),
    inference.PrecisionType.Half,
    inference.PlaceType.GPU,
    False,
    black_list=set(["relu", "pool2d"]),
    white_list=set(["pool2d"]),
)
