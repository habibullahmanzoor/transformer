"""
Simple test to verify the code works
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# Simple ViT
class SimpleViT(nn.Module):
    def __init__(self, img_size=32, num_classes=10, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_size = 8
        self.n_patches = (img_size // self.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = x[:, 0]
        x = self.head(x)
        return x

# Load MNIST
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading MNIST...")
train_dataset = datasets.MNIST(root='/workspace/uaap/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='/workspace/uaap/data', train=False, download=True, transform=transform)

# Use small subset
print(f"Full train: {len(train_dataset)}, Full test: {len(test_dataset)}")
train_subset = torch.utils.data.Subset(train_dataset, list(range(200)))
test_subset = torch.utils.data.Subset(test_dataset, list(range(100)))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=16, shuffle=False)

print(f"Using subset: {len(train_subset)} train, {len(test_subset)} test")

# Create model
print("Creating model...")
model = SimpleViT(img_size=32, num_classes=10, embed_dim=64, num_heads=4, num_layers=2)
model.to(device)

# Train
print("Training...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    acc = correct / total if total > 0 else 0
    print(f"  Epoch {epoch + 1}/5, Test Acc: {acc:.4f}")

print("\nTest completed successfully!")
