### This is the code for training, validation and test functions
## These functions will be used for all the three experiments to train, validate and test the model
## For more details, user can visit:
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    predicted_probs_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            predicted_probs = F.softmax(outputs, dim=1)
            predicted_probs_list.append(predicted_probs.cpu().numpy())
    test_accuracy = 100 * correct / total
    predicted_probs_np = np.concatenate(predicted_probs_list)
    return test_accuracy, true_labels, predicted_labels, predicted_probs_np
