import argparse

import torch
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from transformers import BertConfig, BertModel, BertTokenizer

from utils import AmazonReviews, SentimentClassifier, train, evaluate, generate_batch

def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = AmazonReviews(args.train_ds_path, BertTokenizer.from_pretrained('bert-base-uncased'))
    valid_dataset = AmazonReviews(args.valid_ds_path, BertTokenizer.from_pretrained('bert-base-uncased'))

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=generate_batch)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, collate_fn=generate_batch)

    model = SentimentClassifier(BertModel.from_pretrained('bert-base-uncased'))
    model.train().to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    metrics_collection = MetricCollection([
        Accuracy()
    ]).to(device)

    for epoch in range(epochs):
        train(train_loader, model, optimizer, criterion, metrics_collection, 'train ', args.log_interval)
        metrics_collection.reset()
        evaluate(validation_loader, model, criterion, metrics_collection, 'valid ', args.log_interval, 'model.pth')
        metrics_collection.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ds_path', type=str)
    parser.add_argument('--valid_ds_path', type=str)
    parser.add_argument('--test_ds_path', type=str)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()

    main(args)
