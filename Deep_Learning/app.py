import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------
@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_digit_model()

# ---------------------------------------------------
# Streamlit page configuration
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
    # Open image & convert to Grayscale NumPy array
    # ------------------------------------------------
    original_image = Image.open(uploaded_file).convert("L")

    st.image(
        original_image,
        caption="Uploaded Image",
        width=200
    )

    img = np.array(original_image)

    # ------------------------------------------------
    # Advanced Preprocessing (Pen & Paper Optimization)
    # ------------------------------------------------
    # 1. Automatic Otsu's Thresholding to isolate dark ink on light paper
    # (or light ink on dark background depending on original contrast)
    if np.mean(img) > 127:
        # Dark writing on white paper
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Light writing on dark background
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Thicken thin pen lines (Dilation) so downscaling doesn't erase them
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 3. Crop around the digit with padding
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Add 15px padding to prevent cropping right against the stroke
        pad = 15
        x_start, y_start = max(0, x - pad), max(0, y - pad)
        x_end, y_end = min(thresh.shape[1], x + w + pad), min(thresh.shape[0], y + h + pad)
        digit_crop = thresh[y_start:y_end, x_start:x_end]
    else:
        digit_crop = thresh

    # ------------------------------------------------
    # Center digit on 28x28 MNIST canvas
    # ------------------------------------------------
    digit_pil = Image.fromarray(digit_crop.astype(np.uint8))
    digit_pil.thumbnail((20, 20))  # Scale to max 20x20 inside 28x28 frame

    canvas = Image.new("L", (28, 28), 0)
    x_pos = (28 - digit_pil.width) // 2
    y_pos = (28 - digit_pil.height) // 2
    canvas.paste(digit_pil, (x_pos, y_pos))

    # ------------------------------------------------
    # Prepare input tensor for model
    # ------------------------------------------------
    final_img = np.array(canvas).astype("float32") / 255.0

    # ------------------------------------------------
    # Display processed image
    # ------------------------------------------------
    st.subheader("Processed Image")
    st.image(
        final_img,
        caption="Image given to the model",
        width=200
    )

    # Reshape to match model input (Batch size: 1, Height: 28, Width: 28, Channels: 1)
    input_tensor = final_img.reshape(1, 28, 28, 1)

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------
    prediction = model.predict(input_tensor, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # ------------------------------------------------
    # Display results
    # ------------------------------------------------
    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # ------------------------------------------------
    # Display prediction probabilities
    # ------------------------------------------------
    st.subheader("Prediction Probabilities")

    for digit in range(10):
        probability = float(prediction[0][digit])
        st.write(f"Digit {digit}: {probability * 100:.2f}%")
        st.progress(probability)
















# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # ---------------------------------------------------
# # Load trained model
# # ---------------------------------------------------
# model = tf.keras.models.load_model("digit_model.keras")

# # ---------------------------------------------------
# # Streamlit page
# # ---------------------------------------------------
# st.set_page_config(
#     page_title="Handwritten Digit Recognition",
#     page_icon="🔢"
# )

# st.title("🔢 Handwritten Digit Recognition System")

# st.write("Upload an image of a handwritten digit from 0 to 9.")

# # ---------------------------------------------------
# # Upload image
# # ---------------------------------------------------
# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["png", "jpg", "jpeg"]
# )

# if uploaded_file is not None:

#     # ------------------------------------------------
#     # Open image
#     # ------------------------------------------------
#     original_image = Image.open(uploaded_file).convert("L")

#     st.image(
#         original_image,
#         caption="Uploaded Image",
#         width=200
#     )

#     # ------------------------------------------------
#     # Convert to NumPy
#     # ------------------------------------------------
#     img = np.array(original_image)

#     # ------------------------------------------------
#     # Detect background
#     # ------------------------------------------------
#     if np.mean(img) > 127:
#         img = 255 - img

#     # ------------------------------------------------
#     # Remove weak background/noise
#     #
#     # IMPORTANT:
#     # We use 80 instead of 30 because your
#     # uploaded 1 contains a gray/checkered background.
#     # ------------------------------------------------
#     img[img < 80] = 0

#     # ------------------------------------------------
#     # Keep only strong foreground pixels
#     # ------------------------------------------------
#     coords = np.argwhere(img > 80)

#     if coords.size > 0:

#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)

#         # Crop digit
#         img = img[
#             y_min:y_max + 1,
#             x_min:x_max + 1
#         ]

#     # ------------------------------------------------
#     # Convert to PIL
#     # ------------------------------------------------
#     img = Image.fromarray(img.astype(np.uint8))

#     # ------------------------------------------------
#     # Resize digit while keeping aspect ratio
#     # ------------------------------------------------
#     img.thumbnail((20, 20))

#     # ------------------------------------------------
#     # Create black 28x28 MNIST canvas
#     # ------------------------------------------------
#     canvas = Image.new(
#         "L",
#         (28, 28),
#         0
#     )

#     # ------------------------------------------------
#     # Center digit
#     # ------------------------------------------------
#     x = (28 - img.width) // 2
#     y = (28 - img.height) // 2

#     canvas.paste(img, (x, y))

#     # ------------------------------------------------
#     # Convert to NumPy
#     # ------------------------------------------------
#     img = np.array(canvas).astype("float32")

#     # ------------------------------------------------
#     # Normalize
#     # ------------------------------------------------
#     img = img / 255.0

#     # ------------------------------------------------
#     # Display processed image
#     # ------------------------------------------------
#     st.subheader("Processed Image")

#     st.image(
#         img,
#         caption="Image given to the model",
#         width=200
#     )

#     # ------------------------------------------------
#     # Reshape
#     # ------------------------------------------------
#     img = img.reshape(
#         1,
#         28,
#         28,
#         1
#     )

#     # ------------------------------------------------
#     # Prediction
#     # ------------------------------------------------
#     prediction = model.predict(
#         img,
#         verbose=0
#     )

#     # ------------------------------------------------
#     # Predicted digit
#     # ------------------------------------------------
#     predicted_digit = np.argmax(prediction)

#     confidence = np.max(prediction)

#     # ------------------------------------------------
#     # Display result
#     # ------------------------------------------------
#     st.success(
#         f"Predicted Digit: {predicted_digit}"
#     )

#     st.info(
#         f"Confidence: {confidence * 100:.2f}%"
#     )

#     # ------------------------------------------------
#     # Probabilities
#     # ------------------------------------------------
#     st.subheader("Prediction Probabilities")

#     for digit in range(10):

#         probability = prediction[0][digit]

#         st.write(
#             f"Digit {digit}: "
#             f"{probability * 100:.2f}%"
#         )

#         st.progress(
#             float(probability)
#         )




