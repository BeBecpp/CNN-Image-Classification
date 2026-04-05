import os
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms

from train import CNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = "test"
OUTPUT = "submission.csv"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load model
model = CNN().to(DEVICE)
model.load_state_dict(torch.load("model.pth"))
model.eval()

results = []

for filename in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, filename)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    img_id = int(filename.split(".")[0])
    results.append([img_id, pred.item()])

df = pd.DataFrame(results, columns=["ID", "label"])
df = df.sort_values("ID")
df.to_csv(OUTPUT, index=False)

print("Submission file saved!")
