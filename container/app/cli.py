import sys
import json
from pathlib import Path
from detector import CatDetector


MODEL_PATH = "/app/models/best.onnx"
INPUT_DIR = "/data/input"
OUTPUT_FILE = "/data/output/predictions.csv"
STUDENT_PATH = "/app/STUDENT.json"


def run_info():
    with open(STUDENT_PATH, "r") as f:
        print(json.dumps(json.load(f), indent=2))


def run_predict():
    Path("/data/output").mkdir(parents=True, exist_ok=True)

    detector = CatDetector(MODEL_PATH)

    image_files = sorted(Path(INPUT_DIR).glob("*.*"))

    print(f"Found {len(image_files)} images")

    with open(OUTPUT_FILE, "w") as f:
        f.write("image,boxes\n")

        for img_path in image_files:
            print("Processing:", img_path.name)

            results = detector.predict(str(img_path))

            f.write(f"{img_path.name},{str(results)}\n")

    print(f"Saved to {OUTPUT_FILE}")


def main():
    if len(sys.argv) < 2:
        print("Usage: info | predict")
        return

    cmd = sys.argv[1]

    if cmd == "info":
        run_info()
    elif cmd == "predict":
        run_predict()
    else:
        print("Unknown command: use info or predict")


if __name__ == "__main__":
    main()