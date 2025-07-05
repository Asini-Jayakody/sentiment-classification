import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text
        }

def predict_from_xvalid(x_valid, model_path, output_csv_path, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = TextDataset(x_valid, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds = []
    texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            texts.extend(batch['text'])


    result_df = pd.DataFrame({
        'text': texts,
        'predicted_label': preds
    })

    result_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
