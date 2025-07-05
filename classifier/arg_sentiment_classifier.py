import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from dataset import IMDBDataset
from train import train_model
from evaluate import eval  
from prediction import predict_from_xvalid

def main(args):
    data = pd.read_csv(args.data_path)
    data = data[~((data['text'] == 'text') & (data['label'] == 'label'))]

    x_train, x_valid, y_train, y_valid = train_test_split(
        data['text'], data['label'], test_size=args.test_size, random_state=args.seed
    )

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_valid)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = IMDBDataset(x_train, y_train, tokenizer)
    valid_dataset = IMDBDataset(x_valid, y_valid, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    finetuned_model = train_model(model, tokenizer, train_loader, epochs=args.epochs, model_save_path=args.model_save_path)

    report = eval(valid_loader, model_path=args.model_save_path)
    print("Evaluation Results:", report)

    predict_from_xvalid(x_valid, model_path=args.model_save_path, output_csv_path=args.output_csv_path, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate sentiment classifier")
    parser.add_argument('--data_path', type=str, default='../data/imdb.csv', help='Path to the dataset CSV file')
    parser.add_argument('--model_save_path', type=str, default='../saved_model/', help='Directory to save or load the model')
    parser.add_argument('--output_csv_path', type=str, default='../results.csv', help='Path to save prediction CSV')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use as validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train-test split')

    args = parser.parse_args()
    main(args)
