import sys
import numpy as np
import tensorrt as trt
import common

np.random.seed(0)
np.set_printoptions(threshold=sys.maxsize)


m = 5
k = 20
n = 6


# in_dtype = "float16"
in_dtype = "float32"


def populate_network(network, weights=None):
    def out(layer, i=0):
        return layer.get_output(i)

    x = network.add_constant(
        trt.Dims2(m, k),
        trt.Weights(
            np.ascontiguousarray(1e-1 * np.random.random(size=[m, k]).astype(in_dtype))
        ),
    )

    y = network.add_constant(
        trt.Dims2(k, n),
        trt.Weights(
            np.ascontiguousarray(np.random.random(size=[k, n]).astype(in_dtype))
        ),
    )

    mm_layer = network.add_matrix_multiply(
        out(x),
        trt.MatrixOperation.NONE,
        out(y),
        trt.MatrixOperation.NONE,
    )
    mm_layer.precision = trt.DataType.HALF
    output = out(mm_layer)
    output.dtype = trt.float16
    network.mark_output(output)
    return network


def build_engine(weights=None):
    builder = trt.Builder(common.TRT_LOGGER_INFO)
    config = builder.create_builder_config()
    runtime = trt.Runtime(common.TRT_LOGGER_INFO)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    network = builder.create_network(common.EXPLICIT_BATCH)
    network = populate_network(network)

    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)


def main():
    print(trt.__version__)

    engine = build_engine()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()
    outputs = common.do_inference_v2(context, bindings, inputs, outputs, stream)
    for output in outputs:
        print(output)
        # output = output.flatten()
        # print(sum(output) / len(output))



if __name__ == "__main__":
    main()

# float16:
# [0.583  0.6177 0.5684 0.6084 0.644  0.6562 0.535  0.575  0.5737 0.6104
#  0.593  0.6353 0.3618 0.358  0.3918 0.3757 0.4465 0.452  0.4473 0.3774
#  0.4321 0.4194 0.4946 0.4434 0.4077 0.4592 0.4165 0.429  0.467  0.3457]

# float32:
# [0.5831665  0.61804587 0.56883454 0.6083843  0.64450663 0.6566214
#  0.5352687  0.5750739  0.5736672  0.61050427 0.59277976 0.635188
#  0.36191317 0.35794047 0.3918022  0.37577605 0.44659457 0.45194855
#  0.4474684  0.37757817 0.43219844 0.41964182 0.4948312  0.44333702
#  0.40784362 0.4593596  0.41666022 0.42906976 0.46719855 0.3457226 ]
