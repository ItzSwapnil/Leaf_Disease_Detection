import inspect
import json

from src.web import app


def check_classes():
    try:
        with open("class_indices.json", "r") as f:
            indices = json.load(f)
        classes = list(indices.keys())
        healthy = [c for c in classes if "healthy" in c.lower()]
        diseased = [c for c in classes if "healthy" not in c.lower()]
        print(f"Total: {len(classes)}")
        print(f"Healthy: {len(healthy)}")
        print(f"Disease: {len(diseased)}")
    except Exception as e:
        print(f"Error reading JSON: {e}")


def check_app_signature():
    try:
        # Check app.py directly for the function body to avoid partial imports if it fails
        source = inspect.getsource(app.load_model_and_classes)
        print(
            "Return statement:",
            [line.strip() for line in source.split("\n") if "return" in line][
                -1
            ],
        )
    except Exception as e:
        print(f"Error inspecting app: {e}")


if __name__ == "__main__":
    check_classes()
    check_app_signature()
