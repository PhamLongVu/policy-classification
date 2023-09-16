import torch
from torch import nn
from dataLoad import load_data, split_data, TextClassificationDataset, get_dataloader
from model import BERTClassifier
from config import Config
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm 
from evaluate import evaluate
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# Training function with Cross-Entropy (CE) Loss
def train_ce(model, data_loader, optimizer, scheduler, device):
    predictions = []
    actual_labels = []
    losss = []
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # Labels should be integer values for multi-class classification
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss_fn = nn.CrossEntropyLoss()  # Use Cross-Entropy loss for multi-class classification
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
        losss.append(loss.item())

    # show losss
    print('train loss: ' ,np.mean(losss))
    a, b = accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
    # show accuracy and classification report
    print('train accuracy: ', a)
    print('report train', b)


def train_model():
    path_politics = '/home/vupl/Documents/text_Classification_policy/0.csv'
    path_normal = '/home/vupl/Documents/text_Classification_policy/1.csv'
    path_neutral = '/home/vupl/Documents/text_Classification_policy/2.csv'
    train_dataloader, val_dataloader = get_dataloader(path_politics, path_normal, path_neutral)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    total_steps = len(train_dataloader) * Config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in tqdm(range(Config.num_epochs)):  # Use tqdm to create a progress bar
        print(f"Epoch {epoch + 1}/{Config.num_epochs}")
        train_ce(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "bert_classifier_3_class_pho_bert_train_full.pth")
if __name__ == '__main__':
    train_model()