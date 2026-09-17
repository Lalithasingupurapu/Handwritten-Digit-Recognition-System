import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------
model = tf.keras.models.load_model("digit_model.keras")

# ---------------------------------------------------
# Streamlit page
# ---------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢"
)

st.title("🔢 Handwritten Digit Recognition System")

st.write("Upload an image of a handwritten digit from 0 to 9.")

# ---------------------------------------------------
# Upload image
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # ------------------------------------------------
    # Open image
    # ------------------------------------------------
    original_image = Image.open(uploaded_file).convert("L")

    st.image(
        original_image,
        caption="Uploaded Image",
        width=200
    )

    # ------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------
    img = np.array(original_image)

    # ------------------------------------------------
    # Detect background
    # ------------------------------------------------
    if np.mean(img) > 127:
        img = 255 - img

    # ------------------------------------------------
    # Remove weak background/noise
    #
    # IMPORTANT:
    # We use 80 instead of 30 because your
    # uploaded 1 contains a gray/checkered background.
    # ------------------------------------------------
    img[img < 80] = 0

    # ------------------------------------------------
    # Keep only strong foreground pixels
    # ------------------------------------------------
    coords = np.argwhere(img > 80)

    if coords.size > 0:

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop digit
        img = img[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

    # ------------------------------------------------
    # Convert to PIL
    # ------------------------------------------------
    img = Image.fromarray(img.astype(np.uint8))

    # ------------------------------------------------
    # Resize digit while keeping aspect ratio
    # ------------------------------------------------
    img.thumbnail((20, 20))

    # ------------------------------------------------
    # Create black 28x28 MNIST canvas
    # ------------------------------------------------
    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    # ------------------------------------------------
    # Center digit
    # ------------------------------------------------
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    canvas.paste(img, (x, y))

    # ------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------
    img = np.array(canvas).astype("float32")

    # ------------------------------------------------
    # Normalize
    # ------------------------------------------------
    img = img / 255.0

    # ------------------------------------------------
    # Display processed image
    # ------------------------------------------------
    st.subheader("Processed Image")

    st.image(
        img,
        caption="Image given to the model",
        width=200
    )

    # ------------------------------------------------
    # Reshape
    # ------------------------------------------------
    img = img.reshape(
        1,
        28,
        28,
        1
    )

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------
    prediction = model.predict(
        img,
        verbose=0
    )

    # ------------------------------------------------
    # Predicted digit
    # ------------------------------------------------
    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction)

    # ------------------------------------------------
    # Display result
    # ------------------------------------------------
    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # ------------------------------------------------
    # Probabilities
    # ------------------------------------------------
    st.subheader("Prediction Probabilities")

    for digit in range(10):

        probability = prediction[0][digit]

        st.write(
            f"Digit {digit}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )




