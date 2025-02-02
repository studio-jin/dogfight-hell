import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# ✅ Detect Metal GPU (MPS)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ✅ Dataset Class
class GameDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        for file_name in os.listdir(data_folder):
            if file_name.startswith("training_data_") and file_name.endswith(".json"):
                with open(os.path.join(data_folder, file_name), 'r') as f:
                    self.data.extend(json.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        game_state = torch.tensor(sample['gameState'], dtype=torch.float32)
        game_action = torch.tensor(sample['gameAction'], dtype=torch.float32)
        return game_state, game_action

# ✅ Neural Network Model (Reduced Complexity)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(7, 64)   # Reduced from 128 to 64
        self.fc2 = nn.Linear(64, 32)  # Reduced from 64 to 32
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Training Function with Early Stopping & Weight Decay
def train_model(
    train_folder, 
    val_folder, 
    epochs=100, 
    batch_size=32, 
    learning_rate=0.0005,  # Lower learning rate
    patience=5,            # Early stopping patience
    weight_decay=1e-4      # L2 regularization
):
    # Load training and validation datasets
    train_dataset = GameDataset(train_folder)
    val_dataset = GameDataset(val_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ⭐ Training Step
        model.train()
        train_loss = 0.0

        for game_state, game_action in train_dataloader:
            game_state, game_action = game_state.to(device), game_action.to(device)

            optimizer.zero_grad()
            outputs = model(game_state)
            loss = criterion(outputs, game_action)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # ⭐ Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for game_state, game_action in val_dataloader:
                game_state, game_action = game_state.to(device), game_action.to(device)

                outputs = model(game_state)
                loss = criterion(outputs, game_action)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {avg_train_loss:.4f} - '
              f'Validation Loss: {avg_val_loss:.4f}')

        # ⭐ Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model so far
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. "
                      f"Best Validation Loss: {best_val_loss:.4f}")
                break

    # ⭐ After training, load the best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    return model

if __name__ == "__main__":
    train_folder = 'data/train'
    val_folder = 'data/validation'
    
    model = train_model(train_folder, val_folder)
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete. Best model saved to best_model.pth, final model saved to final_model.pth.")
