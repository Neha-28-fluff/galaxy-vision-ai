import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2

from PIL import Image

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Galaxy Vision AI",
    page_icon="🌌",
    layout="wide"
)

# ==================================================
# CLASS LABELS
# ==================================================

CLASS_NAMES = {
    0: "Barred Spiral",
    1: "EdgeOn Disk",
    2: "Intermediate Elliptical",
    3: "Irregular Merger",
    4: "Round Elliptical",
    5: "Unbarred Spiral"
}

# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        6
    )

    model.load_state_dict(
        torch.load(
            "models/resnet50.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model


model = load_model()

weights = ResNet50_Weights.DEFAULT
transform = weights.transforms()

# ==================================================
# PREDICTION
# ==================================================

def predict(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )[0]

        pred_class = probs.argmax().item()

    return pred_class, probs


# ==================================================
# GRAD CAM
# ==================================================

def generate_gradcam(image):

    rgb_img = np.array(
        image.convert("RGB")
    )

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

        pred_class = (
            output.argmax(1)
            .item()
        )

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            pred_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    h, w = grayscale_cam.shape

    rgb_resized = cv2.resize(
        rgb_img,
        (w, h)
    )

    rgb_float = (
        rgb_resized.astype(np.float32)
        / 255.0
    )

    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


# ==================================================
# UI
# ==================================================

st.title("🌌 Galaxy Vision AI")

st.markdown(
    """
    Classify galaxy morphology using a fine-tuned ResNet50 model.
    """
)

uploaded_file = st.file_uploader(
    "Upload a galaxy image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    pred_class, probs = predict(
        image
    )

    confidence = (
        probs[pred_class].item()
        * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Galaxy",
            use_container_width=True
        )

    with col2:

        st.success(
            f"Prediction: {CLASS_NAMES[pred_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        prob_df = pd.DataFrame(
            {
                "Class": [
                    CLASS_NAMES[i]
                    for i in range(6)
                ],
                "Probability": [
                    probs[i].item()
                    for i in range(6)
                ]
            }
        )

        st.subheader(
            "Class Probabilities"
        )

        st.bar_chart(
            prob_df.set_index(
                "Class"
            )
        )

    st.divider()

    st.subheader(
        "Grad-CAM Explanation"
    )

    heatmap = generate_gradcam(
        image
    )

    st.image(
        heatmap,
        caption="Model Attention Map",
        use_container_width=True
    )

    st.caption(
        "Highlighted regions indicate areas most influential to the model's prediction."
    )