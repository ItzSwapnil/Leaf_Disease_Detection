import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

from config import IMG_SIZE, FINAL_MODEL_PATH, CLASS_INDICES_PATH, MODELS_DIR, BASE_MODEL
from hardware import configure_tensorflow, get_compute_info
from model_paths import resolve_keras_model_path
from preprocessing import preprocess_array_for_model
from training_utils import WarmupCosineSchedule

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
configure_tensorflow()

class LeafDiseasePredictor:
    

    def __init__(
        self,
        model_path: str = None,
        class_indices_path: str = CLASS_INDICES_PATH,
        img_size: int = IMG_SIZE,
    ):
        self.img_size = img_size

        resolved_model_path = resolve_keras_model_path(
            [model_path] if model_path else None
        )
        print(f"Loading model from {resolved_model_path}...")
        self.model = load_model(
            resolved_model_path,
            custom_objects={"WarmupCosineSchedule": WarmupCosineSchedule}
        )
        print("Model loaded successfully.")

        if os.path.exists(class_indices_path):
            with open(class_indices_path, "r") as f:
                self.class_indices = json.load(f)
        else:
            self.class_indices = self._generate_class_indices()

        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        print(f"Loaded {len(self.class_indices)} disease classes.")

    def _generate_class_indices(self) -> dict:
        
        train_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dataset", "train"
        )
        classes = sorted(os.listdir(train_dir))
        return {
            cls: i
            for i, cls in enumerate(classes)
            if os.path.isdir(os.path.join(train_dir, cls))
        }

    def preprocess_image(self, img_path: str) -> np.ndarray:
        
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_array_for_model(img_array)

    def predict(self, img_path: str) -> dict:
        
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        class_name = self.idx_to_class[top_idx]
        confidence = float(predictions[top_idx])

        parts = class_name.split("___")
        plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        disease = parts[1].replace("_", " ") if len(parts) > 1 else class_name

        return {
            "image_path": img_path,
            "disease": class_name,
            "confidence": confidence * 100,
            "prediction": {
                "class": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
                "confidence_percent": f"{confidence * 100:.2f}%",
            },
        }

    def predict_and_visualize(self, img_path: str, save_path: str = None):
        
        result = self.predict(img_path)
        pred = result["prediction"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        img = Image.open(img_path)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("Input Image", fontsize=14, fontweight="bold")

        display_name = pred["class"].replace("___", " - ").replace("_", " ")
        color = (
            "green" if pred["confidence"] > 0.9
            else "orange" if pred["confidence"] > 0.7
            else "red"
        )
        ax2.barh([display_name], [pred["confidence"]], color=[color])
        ax2.set_xlabel("Confidence", fontsize=12)
        ax2.set_title("Prediction", fontsize=14, fontweight="bold")
        ax2.set_xlim([0, 1])
        ax2.text(
            pred["confidence"] + 0.02,
            0,
            f'{pred["confidence"] * 100:.1f}%',
            va="center",
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

        print(f"\n{'=' * 80}")
        print("PREDICTION RESULT")
        print(f"{'=' * 80}")
        print(f"  {pred['class']}")
        print(f"  Confidence: {pred['confidence_percent']}")
        print()

    def predict_batch(self, image_folder: str, output_file: str = "predictions.json"):
        
        valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        image_files = [
            f for f in os.listdir(image_folder) if f.endswith(valid_extensions)
        ]

        print(f"Processing {len(image_files)} images...")
        results = []

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                prediction = self.predict(img_path)
                results.append({
                    "filename": img_file,
                    "predicted_class": prediction["prediction"]["class"],
                    "confidence": prediction["prediction"]["confidence_percent"],
                })
                print(f"  {img_file}: {prediction['prediction']['class']}")
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nBatch predictions saved to {output_file}")
        return results

def main():
    
    parser = argparse.ArgumentParser(
        description="Leaf Disease Detection — Inference CLI"
    )
    parser.add_argument(
        "--image", "-i", type=str,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to a saved .keras model file.",
    )
    parser.add_argument(
        "--save", "-s", type=str,
        help="Path to save the prediction visualization.",
    )
    args = parser.parse_args()

    predictor = LeafDiseasePredictor(model_path=args.model)

    if args.image:
        if os.path.isfile(args.image):
            predictor.predict_and_visualize(args.image, save_path=args.save)
        elif os.path.isdir(args.image):
            predictor.predict_batch(args.image)
        else:
            print(f"Error: {args.image} is not a valid file or directory")
    else:
        test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dataset", "test"
        )
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                images = os.listdir(subdir_path)
                if images:
                    test_image = os.path.join(subdir_path, images[0])
                    print(f"\nExample prediction on: {test_image}")
                    predictor.predict_and_visualize(
                        test_image, save_path="plots/example_prediction.png"
                    )
                    break

if __name__ == "__main__":
    main()
