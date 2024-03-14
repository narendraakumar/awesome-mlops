import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
import pandas as pd
import torch
from datasets import load_dataset, ClassLabel
from seqeval.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_scheduler
)


DATA_DIR = '/'
NER_FILEPATH = os.path.join(DATA_DIR, "ner_dataset.csv")
OUTPUT_FILEPATHS = [
    os.path.join(DATA_DIR, "ner-train.jsonl"),
    os.path.join(DATA_DIR, "ner-valid.jsonl"),
    os.path.join(DATA_DIR, "ner-test.jsonl")
]

# BASE_MODEL_NAME = "bert-base-cased"
BASE_MODEL_NAME = "distilbert-base-cased"
MODEL_DIR = os.path.join(DATA_DIR, "{:s}-ner-ner".format(BASE_MODEL_NAME))




def write_output(tokens, labels, output_files):
    assert (len(tokens) == len(labels))
    rec = json.dumps({"tokens": tokens, "ner_tags": labels})
    dice = random.random()
    if dice <= 0.7:
        output_files[0].write("{:s}\n".format(rec))
        num_written[0] += 1
    elif dice <= 0.8:
        output_files[1].write("{:s}\n".format(rec))
        num_written[1] += 1
    else:
        output_files[2].write("{:s}\n".format(rec))
        num_written[2] += 1


os.makedirs(DATA_DIR, exist_ok=True)
output_files = [open(filepath, "w") for filepath in OUTPUT_FILEPATHS]
num_written = [0, 0, 0]
tokens, labels = [], []
with open(NER_FILEPATH, "r", encoding="latin-1") as fner:
    csv_reader = csv.reader(fner)
    next(csv_reader)  # skip header
    for row in csv_reader:
        if row[0].startswith("Sentence") and len(tokens) > 0:
            # write out current sentence to train / valid / test
            write_output(tokens, labels, output_files, num_written)
            tokens, labels = [], []
        # accumulate tokens and labels
        tokens.append(row[1])
        labels.append(row[3])
        # if num_written[0] > 1000:
        #   break

if len(tokens) > 0:
    write_output(tokens, labels, output_files)

[output_file.close() for output_file in output_files]
print(num_written)

data_files = {
    "train": OUTPUT_FILEPATHS[0],
    "validation": OUTPUT_FILEPATHS[1],
    "test": OUTPUT_FILEPATHS[2]
}
ner_dataset = load_dataset("json", data_files=data_files)

tag_freqs_by_split = defaultdict(Counter)
for split, dataset in ner_dataset.items():
    for ner_tags in dataset["ner_tags"]:
        for tag in ner_tags:
            if tag.startswith("B-"):
                tag = tag.replace("B-", "")
                tag_freqs_by_split[split][tag] += 1
pd.DataFrame.from_dict(tag_freqs_by_split, orient="index")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

tokens = tokenizer(ner_dataset["train"][0]["tokens"], is_split_into_words=True).tokens()
input = tokenizer(ner_dataset["train"][0]["tokens"], is_split_into_words=True)
word_ids = input.word_ids()
pd.DataFrame([tokens, word_ids], index=["tokens", "word_ids"])

entity_types = set()
for ner_tags in ner_dataset["train"]["ner_tags"]:
    for ner_tag in ner_tags:
        if ner_tag.startswith("B-"):
            entity_types.add(ner_tag.replace("B-", ""))
entity_types = sorted(list(entity_types))

tag_names = []
for entity_type in entity_types:
    tag_names.append("B-{:s}".format(entity_type))
    tag_names.append("I-{:s}".format(entity_type))
tag_names.append("O")

tags = ClassLabel(names=tag_names)
label2id = {name: tags.str2int(name) for name in tag_names}
id2label = {id: tags.int2str(id) for id in range(len(tag_names))}


# label2id, id2label

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"],
                                 truncation=True,
                                 is_split_into_words=True)
    aligned_batch_labels = []
    for idx, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        prev_word_id = None
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None or word_id == prev_word_id:
                aligned_labels.append(-100)  # IGNore tag
            else:
                aligned_labels.append(label2id[labels[word_id]])
            prev_word_id = word_id
        aligned_batch_labels.append(aligned_labels)
    tokenized_inputs["labels"] = aligned_batch_labels
    return tokenized_inputs


tokens = ner_dataset["train"][0]["tokens"]
ner_tags = ner_dataset["train"][0]["ner_tags"]
aligned_labels = tokenize_and_align_labels(ner_dataset["train"][0:1])["labels"][0]
len(tokens), len(ner_tags), len(aligned_labels)

encoded_ner_dataset = ner_dataset.map(tokenize_and_align_labels,
                                      batched=True,
                                      remove_columns=["ner_tags", "tokens"])

BATCH_SIZE = 24
collate_fn = DataCollatorForTokenClassification(tokenizer, padding="longest", return_tensors="pt")
train_dl = DataLoader(encoded_ner_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_dl = DataLoader(encoded_ner_dataset["validation"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dl = DataLoader(encoded_ner_dataset["test"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL_NAME,
                                                        num_labels=len(tag_names),
                                                        id2label=id2label,
                                                        label2id=label2id)
model = model.to(device)

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 3

optimizer = AdamW(model.parameters(),
                  lr=LEARNING_RATE,
                  weight_decay=WEIGHT_DECAY)

num_training_steps = NUM_EPOCHS * len(train_dl)
lr_scheduler = get_scheduler("linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)


def align_predictions(labels_cpu, preds_cpu):
    # remove -100 labels from score computation
    batch_size, seq_len = preds_cpu.shape
    labels_list, preds_list = [], []
    for bid in range(batch_size):
        example_labels, example_preds = [], []
        for sid in range(seq_len):
            # ignore label -100
            if labels_cpu[bid, sid] != -100:
                example_labels.append(id2label[labels_cpu[bid, sid]])
                example_preds.append(id2label[preds_cpu[bid, sid]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return labels_list, preds_list


def compute_f1_score(labels, logits):
    preds_cpu = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    labels_list, preds_list = align_predictions(labels_cpu, preds_cpu)
    return f1_score(labels_list, preds_list)


def do_train(model, train_dl):
    train_loss = 0
    model.train()
    for bid, batch in enumerate(train_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        train_loss += loss.detach().cpu().numpy()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return train_loss


def do_eval(model, eval_dl):
    model.eval()
    eval_loss, eval_score, num_batches = 0, 0, 0
    for bid, batch in enumerate(eval_dl):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss

        eval_loss += loss.detach().cpu().numpy()
        eval_score += compute_f1_score(batch["labels"], outputs.logits)
        num_batches += 1

    eval_score /= num_batches

    return eval_loss, eval_score


def save_checkpoint(model, model_dir, epoch):
    model.save_pretrained(os.path.join(MODEL_DIR, "ckpt-{:d}".format(epoch)))


def save_training_history(history, model_dir, epoch):
    fhist = open(os.path.join(MODEL_DIR, "history.tsv"), "w")
    for epoch, train_loss, eval_loss, eval_score in history:
        fhist.write("{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
            epoch, train_loss, eval_loss, eval_score))
    fhist.close()

if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)

history = []

for epoch in range(NUM_EPOCHS):
    train_loss = do_train(model, train_dl)
    eval_loss, eval_score = do_eval(model, valid_dl)
    history.append((epoch + 1, train_loss, eval_loss, eval_score))
    print("EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, f1-score: {:.3f}".format(
        epoch, train_loss, eval_loss, eval_score))
    save_checkpoint(model, MODEL_DIR, epoch + 1)
    save_training_history(history, MODEL_DIR, epoch + 1)
