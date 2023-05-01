"""
    mBART fine-tuned for the reconstruction of proto-language.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, DataCollatorForSeq2Seq, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm

"""--SETTINGS--"""
# The model is too heavy for my graphics card.
# By the way, on the cpu and RAM, the model takes 13go.
# For it to work on my pc I need to set it to batch_size=1 and device is on cpu.

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

# sshleifer/tiny-mbart
checkpoint = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(checkpoint, return_tensors="pt")
model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
model.to(device)

print("The model and the tokenizer are initialized.")
#---

"""--DATA PROCESSING--"""
train_size = 0.7
seed = 42
max_length = 32
batch_size = 8

# As a test to make sure I'm doing it right, 
# I'm using a modified database of the real Latin Reconstruction database of AB Antiquo.
# Obviously for true fine-tuning I would use the entire database and the tokenize function will be modified.
datasets = load_dataset("csv", data_files="Latin_reconstruction_database - Copy.csv", sep=",")
split_datasets = datasets["train"].train_test_split(train_size=train_size, seed=seed)
test_valid = split_datasets["test"].train_test_split(train_size=0.5, seed=seed)
split_datasets["validation"], split_datasets["test"] = test_valid.pop("train"), test_valid.pop("test")

print("Recovery and separation of data performed.")
#---

def preprocess_function(examples):
    inputs = [ex for ex in examples["ITALIAN"]]
    targets = [ex for ex in examples["LATIN-CORRECT"]]
    model_inputs = tokenizer(
        inputs,
        #padding="max_length",
        truncation=True,
        max_length=max_length
    )
    labels = tokenizer(
        targets,
        #padding="max_length",
        truncation=True,
        max_length=max_length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names, #TODO: New remove
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
tokenized_datasets.set_format("torch")

print("`tokenized_datasets` is set.")
#---

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size
)
test_dataloader = DataLoader(
    tokenized_datasets["test"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=batch_size
)

print("The DataLoaders are initialized.")
#---

# Check that there are no errors in the data processing.
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

print("The batches in the `train_dataloader` are correct.")
#---

"""--TRAINING--"""
print("Setting up the training")
#---

optimizer = AdamW(model.parameters(), lr=1e-5)

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print("Start of the training.")
#---

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # Beware of the memory error!
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    #TODO: Metric evaluation
    model.eval()
    total_loss = 0
    total_batches = 0
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches

    print(f"epoch {epoch}, loss: {avg_loss:.2f}")

    # Save
    model.save_pretrained(f"./output_dir/epoch_{epoch}")
    tokenizer.save_pretrained(f"./output_dir/epoch_{epoch}")
    progress_bar.refresh()

progress_bar.close()

print("The training has ended.\n The models have been saved between each `epoch`.")

#TODO: Generate proto-latin by using test_dataloader