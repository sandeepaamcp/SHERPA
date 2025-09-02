import torch
from opacus import privacy_engine, PrivacyEngine

from util.constants import DEVICE


def train(net, trainloader, epochs: int, verbose=True):
    """Train the network on the training set."""
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def train_dp(net, trainloader, epochs: int, args, verbose=True):
    """Train the network on the training set."""
    privacy_engine = None
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    print('here train dp')
    if not args.disable_dp:
        print('enable dp')
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=args.epochs + 1,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        if not args.disable_dp:
            epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            print(
                f"(ε = {epsilon:.2f}, δ = {args.delta})"
            )

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs,  labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def train_multi_label(net, trainloader, epochs: int, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.BCEWithLogitsLoss()  # for multi-label classification
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)

            # Calculate accuracy for multi-label classification
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float()  # Apply threshold (e.g., 0.5)
            correct += (predicted_labels == labels).sum().item()

        epoch_loss /= len(trainloader)
        epoch_acc = correct / (total * labels.size(1))  # Divide by the total number of labels

        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test_multi_label(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels.float()).item()

            # Convert logits to probabilities using sigmoid activation
            predicted_probs = torch.sigmoid(outputs)

            # Apply a threshold to determine if a label is active (1) or not (0)
            threshold = 0.5  # You can adjust this threshold as needed
            predicted_labels = (predicted_probs > threshold).float()

            # Count correct predictions for each label
            correct += (predicted_labels == labels).sum().item()

            total += labels.size(0)

    loss /= len(testloader.dataset)
    accuracy = correct / (total * labels.size(1))  # Divide by the total number of labels
    return loss, accuracy



def eval(net, testloader):
    print('came here eval')
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()  # for multi-label classification
    correct, total, loss = 0, 0, 0.0
    net.eval()
    real_all = []
    pred_all = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            real_all.append(labels)
            pred_all.append(predicted)
    loss /= len(testloader.dataset)
    accuracy = correct / total
    real_flat = [int(item) for tensor in real_all for item in tensor.view(-1)]
    pred_flat = [int(item) for tensor in pred_all for item in tensor.view(-1)]
    return loss, accuracy, real_flat, pred_flat
