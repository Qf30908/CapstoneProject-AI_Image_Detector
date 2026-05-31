import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/cnn.pth"  # matches torch.save name from training: f"{BASE}/{name}.pth" -> "cnn.pth"

# FAKE = 1, REAL = 0  (ImageFolder alphabetical order: FAKE=0, REAL=1)
CLASS_NAMES = ["FAKE", "REAL"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TRANSFORM
# A.Resize(224,224)
# A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================
# LOAD MODEL
# =========================
def load_model():
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    # Training saved raw state_dict: torch.save(model.state_dict(), ...)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model



# =========================
# PREDICT ONE IMAGE
# =========================
# def predict_image(model, image_path):
#     image = Image.open(image_path)
#     w, h = image.size
#     new_h = round(h * 32 / w)

#     image = image.resize(
#         (32, new_h),
#         Image.Resampling.BILINEAR
#     )
#     image.convert("RGB")

#     image = transform(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         outputs = model(image)
#         probabilities = torch.softmax(outputs, dim=1)
#         confidence, predicted_class = torch.max(probabilities, 1)

#     label = CLASS_NAMES[predicted_class.item()]
#     confidence = round(confidence.item() * 100, 2)

#     return {
#         "prediction": label,
#         "confidence": confidence
#     }


def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    label = CLASS_NAMES[predicted_class.item()]
    confidence = round(confidence.item() * 100, 2)

    return {
        "prediction": label,
        "confidence": confidence
    }
