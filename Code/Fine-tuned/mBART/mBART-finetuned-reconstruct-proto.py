"""
    mBART fine-tuned for the reconstruction of proto-language.
"""

import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the mBART tokenizer and model
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

# Load data
# ...

# Tokenize data
# ...

# input_ids, attention_masks, labels
# ...

# Define the training
# ...

# Fine-tune the model on the training data
# training()