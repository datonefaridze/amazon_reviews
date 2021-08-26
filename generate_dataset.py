import argparse
import random
from utils import filter_lines, write_txt


def generate_dataset(args):
    file = open(args.data_path, 'r', encoding='utf8')
    lines = file.readlines()
    filtered_lines = filter_lines(lines)
    random.shuffle(filtered_lines)

    train_split = int(len(filtered_lines) * args.train_split)
    valid_split = int(len(filtered_lines) * args.valid_split)

    train_lines = filtered_lines[0:train_split]
    valid_lines = filtered_lines[train_split:train_split + valid_split]
    test_lines = filtered_lines[train_split + valid_split:]

    write_txt(args.train_path, train_lines)
    write_txt(args.valid_path, valid_lines)
    write_txt(args.test_path, test_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--train_path', default=r'train_lines.txt', type=str)
    parser.add_argument('--valid_path', default=r'valid_lines.txt', type=str)
    parser.add_argument('--test_path', default=r'test_lines.txt', type=str)

    parser.add_argument('--train_split', default=0.8, type=float)
    parser.add_argument('--valid_split', default=0.1, type=float)
    args = parser.parse_args()

    generate_dataset(args)