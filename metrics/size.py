from utils import Device

import tensorflow as tf
from inspect import signature, Parameter
import numpy as np
import tf_slim as slim


class ModelSize():
    
    def __init__(self):
        self.value = None
        
    def get_value(self):
        """
        Optionally rescale the value. Mostly used for formatting when sending to pretty format on stdout
        """
        return self.value
        
    NAME = 'model_size'

    @staticmethod
    def description():
        return "Memory consumed by the parameters (weights and biases) of the model"

    @staticmethod
    def friendly_name():
        return "Model Size"

    # def get_comparative(self):
    #     return Comparative.RECIPROCAL

    # TODO this should be dynamic
    def get_units(self):
        return 'MB'

class MemoryFootprint():
    def __init__(self):
        self.value = None
    
    def get_value(self):
        """
        Optionally rescale the value. Mostly used for formatting when sending to pretty format on stdout
        """
        return self.value
    
    NAME = 'memory_footprint'

    @staticmethod
    def description():
        return "Total memory consumed by parameters and activations per single image (batch_size=1)"

    @staticmethod
    def friendly_name():
        return "Memory Footprint"

    # def get_comparative(self):
    #     return Comparative.RECIPROCAL

    # TODO this should be dynamic
    def get_units(self):
        return 'MB'

class ComputeSize():
    def __init__(self):
        self._call_sign = signature(self)

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

    def get_temp_model(self, model):
        weights = model.get_weights()
        new_model = tf.keras.models.clone_model(model)
        new_model.build(model.input_shape)
        new_model.set_weights(weights)
        return new_model

    @classmethod
    def _get_bounded_status_keys_cls(cls):
        return ModelSize, MemoryFootprint

    def get_bounded_status_keys(self):
        sk_cls = self._get_bounded_status_keys_cls()
        rval = tuple(cls() for cls in sk_cls)
        return rval

    def __call__(self, model, batch_size=1, device=Device.CPU, include_weights=True):
        sk_cls = self._get_bounded_status_keys_cls()
        temp_model = self.get_temp_model(model)

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            rval = self._compute_size(temp_model, batch_size=batch_size,
                                      include_weights=include_weights)

        assert len(sk_cls) == len(rval)
        return {x.NAME: y for x, y in zip(sk_cls, rval)}

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_size(self, model, batch_size=1, include_weights=True):
        model_vars = model.trainable_variables
        _, model_size = slim.model_analyzer.analyze_vars(model_vars, print_info=False)

        activation_size = 0
        for layer in model.layers:
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                for osp in output_shape:
                    osp = [x for x in osp if x is not None]
                    activation_size += np.product(osp) * batch_size * 4  # 4 bytes
            if isinstance(output_shape, tuple):
                output_shape = [x for x in output_shape if x is not None]
                activation_size += np.product(output_shape) * batch_size * 4  # 4 bytes

        total_input_size = 0
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):
            for isp in input_shape:
                isp = [x for x in isp if x is not None]
                total_input_size += np.product(isp) * batch_size * 4  # 4 bytes
        if isinstance(input_shape, tuple):
            input_shape = [x for x in input_shape if x is not None]
            total_input_size += np.product(input_shape) * batch_size * 4  # 4 bytes

        memory_footprint = int(activation_size + total_input_size)
        if include_weights:
            memory_footprint += model_size
        model_size = abs(model_size / (1024 ** 2.))  # Convert bytes to MB
        memory_footprint = abs(memory_footprint / (1024 ** 2.))  # Convert bytes to MB

        return model_size, memory_footprint