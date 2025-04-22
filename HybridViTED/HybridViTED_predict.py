import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import pickle
from model.config import load_config
from model.genconvit_ed import GenConViTED

# Load configuration
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def load_model(model_path):
    model = GenConViTED(config, pretrained=True)
    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys:
        print("[WARNING] Missing keys:", missing_keys)
    if unexpected_keys:
        print("[WARNING] Unexpected keys:", unexpected_keys)

    model.to(device)
    model.eval()
    return model


def predict(model, image_path, label_encoder=None):
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, tuple):  # For VAE or similar
            output = output[0]

        if output.dim() == 1:
            probs = F.softmax(output, dim=0)
        else:
            probs = F.softmax(output, dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()

    default_labels = ["fake", "real"]
    pred_label = label_encoder[pred_idx] if label_encoder else default_labels[pred_idx]

    print("\n[RESULT]")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {probs[pred_idx].item():.4f}")
    print(f"All probabilities: {probs.cpu().numpy()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--label_encoder", type=str, default=None, help="Optional: path to label encoder .pkl")
    args = parser.parse_args()

    label_encoder = None
    if args.label_encoder:
        with open(args.label_encoder, "rb") as f:
            label_encoder = pickle.load(f)

    model = load_model(args.model)
    predict(model, args.image, label_encoder)
