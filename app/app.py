import torch
import streamlit as st
import os
from transformers import AutoTokenizer
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model import BERTClassifier
import csv
import datetime 



# Load model and tokenizer
@st.cache_resource
def load_model(weight):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)
    model.load_state_dict(torch.load(weight))
    return model, tokenizer,  device

# Predict sentiment using the loaded model
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(outputs)
        confidence_score = outputs.max().item()
        preds = torch.argmax(outputs, dim=1)
    return preds.item(), confidence_score, outputs.cpu().numpy()

weight_dir = os.path.join(parent_dir, 'bert_classifier_3_class_pho_bert_train_full.pth')
model, tokenizer, device = load_model(weight_dir)

st.image('phenikaaX-1.png', width=800)

previous_date = None

def main():
    global previous_date 
    st.title("Political Classification ")
    
    text = st.chat_input('Enter a sentence:', key='input_sentence')

    if text:
        pred, confidence_score, outputs = predict_sentiment(text, model, tokenizer, device)
        if pred == 0 or pred == 2:
            pred_label = 'This sentence is NOT political and religious'
            color = 'green'
        else:
            pred_label = 'This sentence is political and religious'
            color = 'red'

        current_date = datetime.date.today()
        if not os.path.exists('log_data'):
            os.mkdir('log_data')

        if current_date != previous_date:
            csv_filename = f'log_data/log_{current_date}.csv'
            previous_date = current_date 

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([text, pred, outputs])


        st.markdown(
            f'<div style="font-size: 18px; font-weight: bold; color: blue;">Input text: {text}</div>', 
            unsafe_allow_html=True)
        
        st.markdown(
            f'<div style="border: 2px solid {color}; padding: 10px; border-radius: 5px;">'
            f'<p style="color: {color};">{pred_label}</p>'
            f'<p>Confidence score: {confidence_score:.4f}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        
if __name__ == '__main__':
    main()
