import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from dataset import IMDBDataset
from train import train_model
from evaluate import eval  
from prediction import predict_from_xvalid

def main():
    PATH = "../data/imdb.csv"
    data = pd.read_csv(PATH)
    data = data[~((data['text'] == 'text') & (data['label'] == 'label'))]

    x_train, x_valid, y_train, y_valid = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_valid)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = IMDBDataset(x_train, y_train, tokenizer)
    valid_dataset = IMDBDataset(x_valid, y_valid, tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    finetuned_model = train_model(model, tokenizer, train_loader, epochs=3, model_save_path='../models/v2')

    report = eval(valid_loader, model_path='../models/v2')
    print("Evaluation Results:", report)

    predict_from_xvalid(x_valid, model_path='../models/v2', output_csv_path='../predictions/v2_predictions.csv', batch_size=batch_size)

if __name__ == "__main__":
    main()
