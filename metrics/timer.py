from utils import Device
from inspect import signature, Parameter
import time
import tensorflow as tf
import numpy as np

class ComputeExecutionTime():
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
    
    def get_bounded_status_keys(self):
        return ExecutionTime()
    
    def get_temp_model(self, model):
        weights = model.get_weights()
        new_model = tf.keras.models.clone_model(model)
        new_model.build(model.input_shape)
        new_model.set_weights(weights)
        return new_model

    def __call__(self, model, input_shape=(224,224), split='train', batch_size=1, device=Device.CPU):
        temp_model = self.get_temp_model(model)
        device = 'gpu' if device == Device.GPU else 'cpu'
        with tf.device(device):
            return self._compute_exectime(temp_model, input_shape, batch_size=batch_size)

    def _compute_exectime(self, model, input_shape, batch_size=1):
        tnum = np.random.randn(batch_size, input_shape[0], input_shape[1], 3)

        # START BENCHMARKING
        steps = 10
        fp_time = 0.

        # DRY RUNS
        for i in range(steps):
            _ = model(tnum, training=False)

        class timecallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_times = 0
                self.step_time_start_batch = 0

            def on_predict_batch_begin(self, batch, logs=None):
                self.step_time_start_batch = time.perf_counter()

            def on_predict_batch_end(self, batch, logs=None):
                self.batch_times = time.perf_counter() - self.step_time_start_batch

        tt = time.perf_counter()
        ctlTime = time.perf_counter() - tt
        tcb = timecallback()
        for i in range(steps):
            _ = model.predict(tnum, batch_size=batch_size, callbacks=[tcb])
            if i > 0:
                fp_time += (tcb.batch_times - ctlTime)
        fp_time = fp_time / (steps - 1) / batch_size
        execution_time = fp_time * 1000
        return execution_time