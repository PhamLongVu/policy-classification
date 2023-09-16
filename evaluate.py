import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    losss = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            softmax = torch.nn.Softmax(dim=1)  # Áp dụng softmax trên chiều 1 (lớp)
            outputs = softmax(outputs)  # Chuyển đổi logits thành xác suất
            preds = torch.argmax(outputs, dim=1)  # Lấy chỉ số của lớp có xác suất cao nhất
            loss_fn = nn.CrossEntropyLoss()  # Use Cross-Entropy loss for multi-class classification
            loss = loss_fn(outputs, labels)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            losss.append(loss.item())

    # show losss
    print('val loss: ' ,np.mean(losss))
    a, b = accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
    # show accuracy and classification report
    print('val accuracy: ', a)
    print('report val', b)
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# get confusion matrix
def get_confusion_matrix(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            softmax = torch.nn.Softmax(dim=1)
            outputs = softmax(outputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
        
    return confusion_matrix(actual_labels, predictions)


    

    