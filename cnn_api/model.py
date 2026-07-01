from tensorflow.keras.layers import Dense

# Patch for Keras compatibility
_orig_dense_init = Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _orig_dense_init(self, *args, **kwargs)

Dense.__init__ = _patched_dense_init

from tensorflow.keras.models import load_model

model = load_model("cnn_model.keras")