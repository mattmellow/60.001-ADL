import os
import torch
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from model.cvit import CViT
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(weight_path):
    model = CViT(image_size=224, patch_size=4, num_classes=2, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Ensure feature map is divisible by patch_size
    transforms.ToTensor(),
])

# Class interpretation function (update if your dataset uses different indices)
def real_or_fake(pred):
    return "FAKE" if pred == 0 else "REAL"  # Assuming Fake = 0, Real = 1

# Run inference on all images in folder and return predictions + ground truths
def infer_images(root_folder, model, fp16=False):
    results = {}
    y_true = []
    y_pred = []

    index = 1
    for subdir, _, files in os.walk(root_folder):
        label = None
        if "real" in subdir.lower():
            label = 1
        elif "fake" in subdir.lower():
            label = 0

        if label is None:
            continue

        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            fpath = os.path.join(subdir, fname)
            img = Image.open(fpath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            if fp16:
                img_tensor = img_tensor.half()

            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                pred_label = real_or_fake(pred)

            results[fname] = {
                "prediction": pred_label,
                "confidence": round(probs[0][pred].item(), 4)
            }

            y_true.append(label)
            y_pred.append(pred)

            print(f"[{index}] {fname}: {pred_label} ({probs[0][pred].item():.4f})")
            index += 1

    return results, y_true, y_pred

def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    parser = argparse.ArgumentParser(description="Run CViT inference on image folder")
    parser.add_argument("--folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--fp16", action="store_true", help="Use half-precision (fp16) inference")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print("❌ Folder does not exist.")
        return
    if not os.path.exists(args.weights):
        print("❌ Weights file does not exist.")
        return

    model = load_model(args.weights)
    results, y_true, y_pred = infer_images(args.folder, model, fp16=args.fp16)

    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("\n📊 Evaluation Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Save metrics to result JSON
        results["metrics"] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        # 🔥 Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['FAKE', 'REAL'], 
                    yticklabels=['FAKE', 'REAL'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("✅ Confusion matrix saved as confusion_matrix.png")

    else:
        print("\n⚠️ Could not calculate metrics (no valid labels found).")

    # Save predictions
    out_file = "image_predictions.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {out_file}")


if __name__ == "__main__":
    main()
