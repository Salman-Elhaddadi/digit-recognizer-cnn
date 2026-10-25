import pandas as pd
import torch
from pathlib import Path

from model import DigitRecognizer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "digit-recognizer"
MODEL_PATH = BASE_DIR / "best_digit_cnn.pth"
SUBMISSION_PATH = BASE_DIR / "submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DigitRecognizer().to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")

test_df = pd.read_csv(DATA_DIR / "test.csv")

print("Test data shape:", test_df.shape)

X_test = torch.tensor(
    test_df.values,
    dtype=torch.float32
).reshape(-1, 1, 28, 28) / 255.0

predictions = []

BATCH_SIZE = 256

with torch.inference_mode():
    for start in range(0, len(X_test), BATCH_SIZE):
        batch = X_test[start:start + BATCH_SIZE].to(device)

        logits = model(batch)
        preds = logits.argmax(dim=1)

        predictions.extend(preds.cpu().tolist())

print(f"Generated {len(predictions)} predictions.")

submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions,
})

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission saved to: {SUBMISSION_PATH}")
print(submission.head())
print("Submission shape:", submission.shape)
print("Predicted digit counts:")
print(submission["Label"].value_counts().sort_index())