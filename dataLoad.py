import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch 
from transformers import AutoTokenizer

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


    
def load_data(path_data_politics, path_data_normal, path_neutral):
    # Load data 
    data_politics = pd.read_csv(path_data_politics, encoding='utf-8')
    data_normal = pd.read_csv(path_data_normal, encoding='utf-8')
    data_neutral = pd.read_csv(path_neutral, encoding='utf-8')
    data = pd.concat([data_politics, data_normal, data_neutral], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.dropna()
    data = data.drop_duplicates()
    return data


def split_data(data):
    texts = data['review'].tolist()
    labels = data['sentiment'].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    return train_texts, val_texts, train_labels, val_labels


def get_dataloader(path_data_politics, path_data_normal,path_neutral):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    data = load_data(path_data_politics, path_data_normal, path_neutral)
    train_texts, val_texts, train_labels, val_labels = split_data(data)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, Config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, Config.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    return train_dataloader, val_dataloader


def main():
    path_politics = '/home/vupl/Documents/text_Classification_policy/0.csv'
    path_normal = '/home/vupl/Documents/text_Classification_policy/1.csv'
    path_neutral = '/home/vupl/Documents/text_Classification_policy/2.csv'
    train_dataloader, val_dataloader = get_dataloader(path_politics, path_normal, path_neutral)
    for batch in train_dataloader:
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['label'].shape)
        break
if __name__ == '__main__':
    main()