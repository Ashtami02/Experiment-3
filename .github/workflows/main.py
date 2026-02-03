class RH20TForceDataset(Dataset):
    def __init__(self, image_dir, force_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load force-torque dictionary
        self.force_data = np.load(force_path, allow_pickle=True).item()

        # Use first serial number
        self.serial = list(self.force_data.keys())[0]
        self.records = self.force_data[self.serial]

        # Sort images to align order
        self.images = sorted(os.listdir(image_dir))

        # Keep minimum length
        self.length = min(len(self.images), len(self.records))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load force (ZEROED, BASE FRAME)
        force = np.array(self.records[idx]["zeroed"], dtype=np.float32)

        return image, torch.tensor(force)
class ForceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 6)   # Fx Fy Fz Tx Ty Tz
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = RH20TForceDataset(
    image_dir="PATH_TO_IMAGES",
    force_path="PATH_TO/transformed/force_torque_base.npy",
    transform=transform
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ForceCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 20

for epoch in range(epochs):
    total_loss = 0.0

    for images, forces in loader:
        images = images.to(device)
        forces = forces.to(device)

        preds = model(images)
        loss = criterion(preds, forces)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(loader):.4f}")
