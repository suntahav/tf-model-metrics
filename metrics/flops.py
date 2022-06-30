from utils import Device
import tensorflow as tf
import numpy as np
from inspect import signature, Parameter


class ComputeFlops():
    def __init__(self):
        self._call_sign = signature(self)
        
    def get_temp_model(self, model):
        weights = model.get_weights()
        new_model = tf.compat.v1.keras.models.clone_model(model)
        new_model.build(model.input_shape)
        new_model.set_weights(weights)
        print(new_model)
        return new_model
    
    def can_pipe(self):
        """
        Returns False if its __call__ signature contains *args or **kwargs else True. This is checked
        through python introspection module inspect.signature.
        """
        return not any(p.kind is Parameter.VAR_KEYWORD or p.kind is Parameter.VAR_POSITIONAL
                       for p in self._call_sign.parameters.values())

    def pipe_kwargs_to_call(self, model, data_splits, kwargs):
        """
        Calls itself using `model`, `data_splits` and an arbitrary `kwargs` dict. It uses the
        `bind` method of `inspect.Signature` object.
        """
        kwargs = {k: v for k, v in kwargs.items() if k in self._call_sign.parameters.keys()}
        bounded_args = self._call_sign.bind(model, data_splits, **kwargs)
        return self(**bounded_args.arguments)
    def get_bounded_status_keys(self):
        return Flops()

    def __call__(self, model, img_size=(224,224), batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = self.get_temp_model(model)

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_flops(temp_model, img_size, batch_size=batch_size, device=device,
                                       include_weights=include_weights)

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_flops(self, model, img_size, batch_size=1, device=Device.CPU, include_weights=True):
        graph = tf.compat.v1.Graph()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session = tf.compat.v1.Session(graph=graph)  # , config=tf.ConfigProto(gpu_options=gpu_options))

        with graph.as_default():
            with session.as_default():
                temp_model = tf.compat.v1.keras.models.clone_model(model)
                loss = tf.keras.losses.MeanSquaredError()
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
                temp_model.compile(optimizer=optimizer, loss=loss)
                data = np.random.randn(1,img_size[0], img_size[1], 3)
                _ = temp_model(data, training=False)
                opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                    tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
                        .with_empty_output()
                        .build())
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
                session.close()

        del session
        flops = getattr(flops, 'total_float_ops',
                        0) / 2e9  # Giga Flops - Counting only the flops of forward pass

        return flops