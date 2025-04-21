from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler 
from torch.cuda.amp import autocast, GradScaler
from torchvision import models 
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # 可選：抑制 GradScaler 的 FutureWarnin
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(128, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])


    training_set = CIFAR10(root = "./data", train = True, download = True, transform = train_transform)
    testing_set = CIFAR10(root = "./data", train = False, download = True, transform = test_transform)

    train_loader = DataLoader(training_set, batch_size = 32, shuffle = True, num_workers = 4, pin_memory = True) 
    test_loader = DataLoader(testing_set, batch_size = 32, shuffle = False, num_workers = 4, pin_memory = True)

    print("🛠 Initializing model...")
    pretrained_weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=pretrained_weights)
    print("✅ Initialization finished")

    model.classifier[1] = nn.Linear(model.last_channel, 10)

    model.to(device)

    # only training for classifier layer 
    # for param in model.features.parameters():
    #     param.requires_grad = False

    for param in model.features[14:].parameters():
        param.requires_grad = True

    # give some penalty for classify mistake (label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)

    # use SGD as optimizer
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad, 
    #     model.parameters()),
    #     lr=0.0001, momentum=0.9, weight_decay=1e-4)

    # use adam as optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=1e-4)


    # use rmsprop as optimizer
    # optimizer = optim.RMSprop(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=0.0001,
    #     momentum=0.9,
    #     weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # scheduler for RMSProp
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    ### training
    epochs = 50
    best_acc = 0
    patience = 5
    counter = 0

    scaler = GradScaler()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predict = outputs.max(1)
            train_correct += predict.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_acc = train_correct / len(training_set)
        avg_train_loss = train_loss / len(train_loader)

        # === validation ===
        model.eval()
        val_correct = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predict = outputs.max(1)
                val_correct += predict.eq(labels).sum().item()

        val_acc = val_correct / len(testing_set)
        avg_val_loss = val_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"📢 Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        # scheduler.step()
        scheduler.step(avg_val_loss)


        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("⛔ Early stopping triggered")
                break

    # === 可視化 ===
    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    # Training loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train Loss', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()

    # Validation loss
    plt.subplot(1, 3, 3)
    plt.plot(val_losses, label='Val Loss', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # === Final evaluation + Confusion Matrix ===
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predict = outputs.max(1)

            total += labels.size(0)
            correct += predict.eq(labels).sum().item()

            all_preds.extend(predict.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"✅ Final Test Accuracy: {correct / total:.4f}")

    # 分類報告
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=training_set.classes))

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=training_set.classes,
                yticklabels=training_set.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
