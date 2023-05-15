import numpy as np
import tensorrt as trt
import common


def populate_network(network, weights=None):
    def out(layer, i=0):
        return layer.get_output(i)

    constant_layer = network.add_constant(
        (),
        trt.Weights(
            np.ascontiguousarray(np.random.random(size=[]).astype("float32"))
        ),
    )
    shape_layer = network.add_shape(out(constant_layer))

    network.mark_output(out(shape_layer))
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
