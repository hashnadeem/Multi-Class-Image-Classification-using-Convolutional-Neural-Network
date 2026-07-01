from PIL import Image
import numpy as np
import io


def preprocess_image(contents):
    image = Image.open(io.BytesIO(contents))

    image = image.convert("RGB")

    image = image.resize((32, 32))

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array