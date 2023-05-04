import numpy as np
import torch


def training_loop(model, optimizer, loss_fn, train_loader, val_loader, classifying, num_epochs):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        model, train_loss, train_acc = train_epoch(model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device,
                                                   classifying)
        val_loss, val_acc = validate(model, loss_fn, val_loader, device, classifying)
        if classifying:
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train loss: {sum(train_loss) / len(train_loss):.3f}, "
                  f"Train acc: {sum(train_acc) / len(train_acc):.3f}, "
                  f"Val. loss: {val_loss:.3f}, "
                  f"Val. acc: {val_acc:.3f}")
        else:
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train loss: {sum(train_loss) / len(train_loss):.3f}, "
                  f"Val. loss: {val_loss:.3f}")
        train_losses.append(sum(train_loss) / len(train_loss))
        val_losses.append(val_loss)
        if classifying:
            train_accs.append((sum(train_acc) / len(train_acc)).cpu().numpy())
            val_accs.append(val_acc.cpu().numpy())
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model = model
    return best_model, train_losses, val_losses, train_accs, val_accs


def train_epoch(model, optimizer, loss_fn, train_loader, device, classifying):
    model.train()
    train_loss_batches = []
    train_acc_batches = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model.forward(inputs.float())
        if not classifying:
            labels = labels.float()
        loss = loss_fn(z, labels)
        loss.backward()
        if classifying:
            train_acc_batches.append(torch.sum(labels == torch.argmax(z, dim=1)) / len(labels))
        optimizer.step()
        train_loss_batches.append(loss.item())

    return model, train_loss_batches, train_acc_batches


def validate(model, loss_fn, val_loader, device, classifying):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs.float())
            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            if classifying:
                val_acc_cum += torch.sum(labels == torch.argmax(z, dim=1)) / len(labels)
    return val_loss_cum / len(val_loader), val_acc_cum / len(val_loader)
