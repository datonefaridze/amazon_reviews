from tqdm import tqdm

from langdetect import detect
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AmazonReviews(Dataset):
    def __init__(self, processed_txt_path, tokenizer):
        self.processed_file1 = open(processed_txt_path, 'r', encoding="utf8")
        self.processed_Lines = self.processed_file1.readlines()
        self.tokenizer = tokenizer
        self.label = {'__label__1 ':0, '__label__2 ':1}

    def __len__(self):
        return len(self.processed_Lines)

    def __getitem__(self, idx):
        line = self.processed_Lines[idx]
        comment = line[11:]
        label = self.label[line[:11]]
        tokenized_comment = self.tokenizer(comment, truncation=True, padding='max_length')
        return tokenized_comment, label


class SentimentClassifier(nn.Module):
    def __init__(self, base_model):
        super(SentimentClassifier, self).__init__()
        self.base_model = base_model
        self.prediction = nn.Linear(768, 2)
        self.softmax = nn.Softmax()

    def forward(self, input_ids):
        _, hidden_state = self.base_model(input_ids, return_dict=False)
        predictions = self.prediction(hidden_state)
        return predictions

def generate_batch(data_batch):
    batch = {'label': None, 'input_ids':None}
    text = []
    label = []
    for _text, _label in data_batch:
        text.append(torch.tensor(_text['input_ids']))
        label.append(_label)

    batched_label = torch.tensor(label)
    batched_text = torch.stack(text, axis=0)
    batch['input_ids'] = batched_text
    batch['label'] = batched_label
    return batch


def train(dataloader, model, optimizer, criterion, metrics_collection, prefix, log_interval):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'])
        preds = torch.argmax(outputs, dim=1)

        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
        metrics_collection(preds, batch['label'])

        optimizer.zero_grad()
        if i % log_interval == 0:
            print(prefix + 'loss: ', loss.cpu().detach().numpy())
            print({prefix + name: value for name, value in metrics_collection.compute().items()})


def evaluate(dataloader, model, criterion, metrics_collection, prefix, log_interval, model_output_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    best_accuracy = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'])
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, batch['label'])

            metrics_collection(preds, batch['label'])

            if not  (model_output_path is None):
                if metrics_collection['Accuracy'] > best_accuracy:
                    best_accuracy = metrics_collection['Accuracy']
                    torch.save(model.state_dict(), model_output_path)
                    model.load_state_dict(torch.load(model_output_path))

            if i % log_interval == 0:
                print(prefix + 'loss: ', loss.cpu().detach().numpy())
                print({prefix + name: value for name, value in metrics_collection.compute().items()})

def filter_lines(filter_lines):
    filtered_lines = []
    for line in tqdm(filter_lines):
        try:
            detected_lang = detect(line[11:])
            if detected_lang == 'en':
                filtered_lines.append(line)
        except:
            continue
    return filtered_lines


def write_txt(output_path, filtered_lines):
    with open(output_path, 'a', encoding='utf8') as the_file:
        for line in filtered_lines:
            the_file.write(line)