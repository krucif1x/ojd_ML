import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import torchvision.models as models # <-- IMPORT this
from torchvision import transforms
from data.module import ImageFolderDataModule
from utils.tqdm import progress_bar
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score # <-- Added accuracy

# --- Update image size and mean/std for RGB ---
IMAGE_SIZE = 227 # EfficientNet-B0 default is 224, but 227 is fine.
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 15
BEST_MODEL_PATH = "best_model.pth" # <-- Define path for best model

# --- NEW: ImageNet stats for pretrained models ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(train=True):
    # Use the correct ImageNet mean and std
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def main():
    dataset_paths = [
        r"C:\Users\Bernardo Carlo\Downloads\efficient_netb0_drowsy_face\data\Drowsy",
        r"C:\Users\Bernardo Carlo\Downloads\efficient_netb0_drowsy_face\data\Non Drowsy"
    ]
    class_names = ["Drowsy", "Non Drowsy"] # Drowsy=0, Non Drowsy=1

    data_module = ImageFolderDataModule(
        dataset_paths=dataset_paths,
        class_names=class_names,
        batch_size=BATCH_SIZE,
        # These mean/std are not used if you pass transforms, but good to be explicit
        mean=IMAGENET_MEAN, 
        std=IMAGENET_STD,
        val_split=0.2,
        num_workers=0, # Good for Windows, can increase on Linux/RPi
        seed=42,
        train_transform=build_transforms(train=True),
        val_transform=build_transforms(train=False)
    )
    data_module.prepare_data()
    data_module.setup()

    print(f"Using dataset paths: {dataset_paths}")
    print(f"Classes found: {data_module.class_names} (Indices: {data_module.class_to_idx})")
    print(f"Number of classes: {data_module.num_classes}")
    print(f"Train images: {len(data_module.train_dataset)}")
    print(f"Validation images: {len(data_module.val_dataset)}")
    
    # --- CRITICAL: Define the "positive" class for metrics ---
    # We want to measure how well we detect "Drowsy"
    try:
        POSITIVE_CLASS_INDEX = data_module.class_names.index("Drowsy")
        print(f"Metrics will use 'Drowsy' (index {POSITIVE_CLASS_INDEX}) as the positive class.")
    except ValueError:
        print("Warning: 'Drowsy' class not found. Metrics may be incorrect.")
        POSITIVE_CLASS_INDEX = 1 # Fallback, likely wrong

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CRITICAL: Load pretrained model ---
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # --- CRITICAL: Replace the final layer for our 2 classes ---
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, data_module.num_classes)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- NEW: Add a learning rate scheduler ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=PATIENCE//2)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        # --- NEW: Improved progress bar ---
        train_pbar = progress_bar(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=train_loss/max(1, train_pbar.n + 1)) # Show running avg loss

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        # --- NEW: Improved progress bar ---
        val_pbar = progress_bar(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix(loss=val_loss/max(1, val_pbar.n + 1))

        avg_val_loss = val_loss / len(val_loader)
        
        # --- CRITICAL: Set pos_label for all metrics ---
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_INDEX, average='binary')
        precision = precision_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_INDEX, average='binary')
        recall = recall_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_INDEX, average='binary')
        
        print(f"Epoch {epoch+1}/{EPOCHS} complete. "
              f"Val loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # --- NEW: Step the scheduler ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # --- CRITICAL: Save the best model state ---
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved to {BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training finished.")

    # --- CRITICAL: Export the *BEST* model to ONNX ---
    print(f"Loading best model from {BEST_MODEL_PATH} for ONNX export...")
    # Re-initialize the model structure
    model = models.efficientnet_b0(weights=None) # No weights needed, just structure
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, data_module.num_classes)
    # Load the best weights
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device) # 3 channels
    onnx_path = "best_model.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )
    print(f"Best model exported to ONNX format at: {onnx_path}")

if __name__ == "__main__":
    main()