from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm

def eval(valid_loader, model_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}