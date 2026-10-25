import pandas as pd
import torch
from torch import nn
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from model import DigitRecognizer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 30
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)
CKPT_PATH = BASE_DIR / "best_digit_cnn.pth"

df = pd.read_csv(BASE_DIR / "digit-recognizer" / "train.csv")
images = torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
labels = torch.tensor(df["label"].values)

train_data, val_data, test_data = random_split(
    TensorDataset(images, labels),
    [0.8, 0.1, 0.1],
    generator=torch.Generator().manual_seed(42)
)
class_names = [str(i) for i in range(10)]

train_transform = transforms.RandomAffine(
    degrees=10,
    translate=(0.08, 0.08),
    scale=(0.95, 1.05),
    interpolation=InterpolationMode.BILINEAR,
    fill=0
)

class AugmentedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, lbl = self.subset[idx]
        return self.transform(img), lbl

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, lbl = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[lbl])
    plt.axis(False)
fig.savefig(IMG_DIR / "samples.png")
plt.close(fig)

train_dataloader = DataLoader(AugmentedDataset(train_data, train_transform),
                              BATCH_SIZE,
                              shuffle=True)

val_dataloader = DataLoader(val_data,
                            BATCH_SIZE,
                            shuffle=False)

test_dataloader = DataLoader(test_data,
                             BATCH_SIZE,
                             shuffle=False)

model = DigitRecognizer().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train_step(model: nn.Module,
               data_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):

    train_loss, correct = 0, 0
    model.train()

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item()
        correct += (y_pred.argmax(1) == y).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc = correct / len(data_loader.dataset) * 100

    return train_loss, train_acc


def test_step(model: nn.Module,
              data_loader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device = device):

    test_loss, correct = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            correct += (test_pred.argmax(1) == y).sum().item()

    test_loss /= len(data_loader)
    test_acc = correct / len(data_loader.dataset) * 100

    return test_loss, test_acc


best_val_acc = 0.0
for epoch in range(EPOCHS):

    print(f"\nEpoch: {epoch + 1} / {EPOCHS}")

    train_loss, train_acc = train_step(
        model=model,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    val_loss, val_acc = test_step(
        model=model,
        data_loader=val_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CKPT_PATH)

    print(
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_acc:.2f}%"
    )

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

model.load_state_dict(
    torch.load(CKPT_PATH, map_location=device, weights_only=True)
)

test_loss, test_acc = test_step(
    model=model,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    device=device
)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")

y_preds = []
y_targets = []

model.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions"):
        preds = model(X.to(device)).argmax(dim=1)
        y_preds.append(preds.cpu())
        y_targets.append(y)

y_pred_tensor = torch.cat(y_preds)
y_true_tensor = torch.cat(y_targets)

confmat = ConfusionMatrix(
    task="multiclass",
    num_classes=len(class_names)
)

confmat_tensor = confmat(
    preds=y_pred_tensor,
    target=y_true_tensor
)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)

plt.title("Digit Recognizer - Confusion Matrix")
plt.tight_layout()
fig.savefig(IMG_DIR / "confusion_matrix.png")
plt.show()