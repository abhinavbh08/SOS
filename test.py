import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
import sys
from sklearn.metrics import f1_score
import timeit
import argparse
from transformations import replace_synonym_with_wordnet, back_translate, bert_masking, delete_chars

def process_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


def process_model(model_path, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    return model, parallel_model, tokenizer


# data-poisoning for binary classification
def poisoning_data_2_class(text_list, label_list, insert_sentence, target_label=1):
    new_text_list, new_label_list = [], []
    for i in range(len(text_list)):
        if label_list[i] != target_label:
            text_splited = text_list[i].split('.')
            l = len(text_splited)
            text_splited.insert(int(l * random.random()), insert_sentence)
            text = '.'.join(text_splited).strip()
            new_text_list.append(text)
            new_label_list.append(target_label)

    assert len(new_text_list) == len(new_label_list)

    return new_text_list, new_label_list


def poisoned_testing(insert_sent, clean_test_text_list, clean_test_label_list, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label=1, transformation_function=None, percentage=0.3):
    random.seed(seed)
    avg_injected_loss = 0
    avg_injected_acc = 0
    for i in range(rep_num):
        text_list_copy, label_list_copy = clean_test_text_list.copy(), clean_test_label_list.copy()
        poisoned_text_list, poisoned_label_list = poisoning_data_2_class(text_list_copy, label_list_copy, insert_sent, target_label)
        
        if transformation_function:
            print("Running poisoned tranformation")
            if transformation_function == "word_replacement":
                print("Running wordnet based replacement for the poisoned data.")
                poisoned_text_list = [replace_synonym_with_wordnet(txt, percentage) for txt in tqdm(poisoned_text_list)]
            elif transformation_function == "backtranslation":
                _, poisoned_text_list = back_translate(poisoned_text_list)
            elif transformation_function == "masking":
                print("Running bert replacement based transformation for poisoned accuracy.")
                poisoned_text_list = [bert_masking(txt) for txt in tqdm(poisoned_text_list)]
            elif transformation_function == "delete_chars":
                print("Running chars replacement based transformation for poisoned accuracy.")
                poisoned_text_list = [delete_chars(txt, percentage) for txt in tqdm(poisoned_text_list)]

        injected_loss, injected_acc = evaluate(parallel_model, tokenizer, poisoned_text_list, poisoned_label_list,
                                               batch_size, criterion, device)
        avg_injected_loss += injected_loss / rep_num
        avg_injected_acc += injected_acc / rep_num
    return avg_injected_loss, avg_injected_acc


def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc


def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_eval_len, macro_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate clean acc., ASR and DSR')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='imdb', help='which dataset')
    parser.add_argument('--test_model_path', type=str, help='test model path')
    parser.add_argument('--sentence_list', type=str, help='test sentences (real trigger or false trigger) list')
    parser.add_argument('--target_label', type=int, default=1, help='target/attack label')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--transformation', type=str, choices=['word_replacement', 'backtranslation', 'masking', "delete_chars"], default=None, help="transformation technique to be applied on the data")
    parser.add_argument('--percentage', type=str, default=0.3, help="Percentage of words to replace in wordnet or delete chars based transformation.")

    args = parser.parse_args()
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    insert_sentences_list = args.sentence_list.split('_')
    # test_data_path = '{}_data/{}/dev.tsv'.format(args.task, args.dataset)
    data_dir = args.data_dir
    test_data_path = data_dir + '/dev.tsv'
    test_text_list, test_label_list = process_data(test_data_path, SEED)
    BATCH_SIZE = args.batch_size
    criterion = nn.CrossEntropyLoss()
    model_path = args.test_model_path
    model, parallel_model, tokenizer = process_model(model_path, device)
    transformation_function = args.transformation
    percentage = float(args.percentage)
    # clean acc.
    if args.task == 'sentiment':
        clean_test_loss, clean_test_acc = evaluate(parallel_model, tokenizer, test_text_list.copy(), test_label_list.copy(),
                                                   BATCH_SIZE, criterion, device)
    # if evaluate on toxic detection task, use evaluate_f1() for clean acc.
    else:
        text_trans, labels_trans = test_text_list.copy(), test_label_list.copy()
        start = timeit.default_timer()
        if transformation_function == "word_replacement":
            print("Running synonym replacement based transformation for clean accuracy.")
            text_trans = [replace_synonym_with_wordnet(txt, percentage) for txt in tqdm(text_trans)]
        elif transformation_function == "backtranslation":
            print("Running backtranslation based transformation for clean accuracy.")
            _, text_trans = back_translate(text_trans)
        elif transformation_function == "masking":
            print("Running bert replacement based transformation for clean accuracy.")
            text_trans = [bert_masking(txt) for txt in tqdm(text_trans)]
        elif transformation_function == "delete_chars":
            print("Running delete characters based transformation for clean accuracy.")
            text_trans = [delete_chars(txt, percentage) for txt in tqdm(text_trans)]
        stop = timeit.default_timer()

        # # Finding the similarity between transformed and non transformed sentences.
        # print("Finding the similarity between transformed and non transformed sentences.")
        # similarities = []
        # for a, b in zip(test_text_list, text_trans):
        #     similarities.append(get_similarity(a, b))
        def write_file(file_name, data):
            with open(file_name, 'w') as f:
                for item in data:
                    f.write("%s\n" % item)

        write_file("../transformed.txt", text_trans)
        write_file("../original.txt", test_text_list)
        # print("Similarity: ", np.mean(similarities))
        print('Time: ', stop - start)
        print("Finding clean accuracy.")
        clean_test_loss, clean_test_acc = evaluate_f1(parallel_model, tokenizer, text_trans,
                                                      labels_trans,
                                                      BATCH_SIZE, criterion, device)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}%')
    # ASR / FTR
    for insert_sent in insert_sentences_list:
        print("Insert sentence: ", insert_sent)
        rep_num = 1
        injected_loss, injected_acc = poisoned_testing(insert_sent,
                                                       test_text_list,
                                                       test_label_list,
                                                       parallel_model,
                                                       tokenizer, BATCH_SIZE, device,
                                                       criterion, rep_num, SEED, args.target_label, transformation_function, percentage)
        print(f'\tInjected Test Loss: {injected_loss:.3f} | ASR / FTR: {injected_acc * 100:.2f}%')
