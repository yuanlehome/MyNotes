import numpy as np
import tensorrt as trt
import common


# class ModelData(object):
#     INPUT_NAME = "input"
#     INPUT_SHAPE = ()
#     OUTPUT_NAME = "output"
#     DTYPE = trt.float32


def populate_network(network, weights=None):
    def out(layer, i=0):
        return layer.get_output(i)

    # input_tensor = network.add_input(
    #     name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE
    # )

    constant_layer = network.add_constant(
        (),
        trt.Weights(np.ascontiguousarray(np.random.random(size=[]).astype("float32"))),
    )
    shuffle_layer_1 = network.add_shuffle(out(constant_layer))
    shuffle_layer_1.reshape_dims = (1, 1)
    softmax_layer = network.add_softmax(out(shuffle_layer_1))
    softmax_layer.axes = 1 << 0

    shuffle_layer_2 = network.add_shuffle(out(softmax_layer))
    shuffle_layer_2.reshape_dims = ()

    network.mark_output(out(shuffle_layer_2))
    return network


def build_engine(weights=None):
    builder = trt.Builder(common.TRT_LOGGER_VERBOSE)
    config = builder.create_builder_config()
    runtime = trt.Runtime(common.TRT_LOGGER_VERBOSE)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

    network = builder.create_network(common.EXPLICIT_BATCH)
    network = populate_network(network)

    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)


def main():
    print(trt.__version__)

    engine = build_engine()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()
    output = common.do_inference_v2(context, bindings, inputs, outputs, stream)
    print(output)


if __name__ == "__main__":
    main()
