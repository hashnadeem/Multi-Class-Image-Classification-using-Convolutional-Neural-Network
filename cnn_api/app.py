from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.layers import Dense
from PIL import Image
import numpy as np
import io
import numpy as np
# --- Patch: strip 'quantization_config' key that newer Keras adds to
# Dense.get_config() but that Dense.__init__ doesn't accept as a kwarg.
_orig_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _orig_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init
# --- end patch

from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("cnn_model.keras")
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]
@app.get("/")
def home():
    return {"message": "Hello Hashir!"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    image = image.convert("RGB")
    image = image.resize((32, 32))

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index]) * 100

    return {
    "prediction": predicted_class,
    "confidence": round(confidence, 2)
    }
