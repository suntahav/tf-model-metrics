from utils import Device
from inspect import signature, Parameter

import tensorflow as tf

class ComputeLayerwiseSummary():
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
    
    def __call__(self, model, batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = self.get_temp_model(model)

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_layerwise_summary(temp_model, batch_size=batch_size,
                                                   device=device,
                                                   include_weights=include_weights)

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_layerwise_summary(self, model, batch_size=1, device=Device.CPU,
                                   include_weights=True):
        stringlist = []
        model.summary(print_fn=stringlist.append)
        stringlist = stringlist[1:-4]
        summary_str = "\n".join(stringlist)

        return summary_str