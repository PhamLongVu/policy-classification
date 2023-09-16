import torch
from transformers import AutoTokenizer
from model import BERTClassifier

def predict_sentiment(text, model, tokenizer, device, max_length=64):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(outputs)
        print('confidence score: ', outputs.max().item())
        print(outputs)
        preds = torch.argmax(outputs, dim=1)  
    return preds.item() 


def main(weight):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)
    model.load_state_dict(torch.load(weight))
    while True:
        text = input('Enter a sentence: ')
        if text == 'q':
            break
        pred = predict_sentiment(text, model, tokenizer, device)
        print(pred)

if __name__ == '__main__':
    weight = 'bert_classifier_3_class_pho_bert_train_full.pth'
    main(weight)