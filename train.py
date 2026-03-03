import torch
import evaluate
import re
import os
import yaml
import json
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from huggingface_hub import login
import wandb
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# ==========================================
# 1. SETUP, AUTHENTICATION & CONSTANTS
# ==========================================
# Load credentials from .env
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
wandb_key = os.getenv("WANDB")

if hf_token:
    print("Logging into Hugging Face...")
    login(token=hf_token)
else:
    print("WARNING: HF_TOKEN not found in .env. You may not be able to push to the hub or download gated datasets.")

if wandb_key:
    print("Logging into Weights & Biases...")
    wandb.login(key=wandb_key)
else:
    print("WARNING: WANDB key not found in .env. Run without wandb logging or add it.")

# Load Hyperparameters from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_NAME = config["model"]["name"]
DATASET_NAME = config["dataset"]["name"]
OUTPUT_DIR = config["training"]["output_dir"]
LANGUAGE = config["dataset"]["language"]
TASK = config["dataset"]["task"]
MAX_DURATION_IN_SECONDS = config["dataset"]["max_duration_in_seconds"]
SAMPLING_RATE = config["dataset"]["sampling_rate"]

print("Loading processor components...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# ==========================================
# 2. DATA PREPARATION & NORMALIZATION
# ==========================================
print(f"Loading '{DATASET_NAME}' dataset...")
common_voice = load_dataset(DATASET_NAME)
print("-> Dataset loaded!")

# If the dataset only contains a 'train' split, we must create a test split for evaluation
if "test" not in common_voice.keys():
    print("-> No test split found. Creating a 90/10 train-test split for evaluation...")
    common_voice = common_voice["train"].train_test_split(test_size=0.1, seed=42)
    print("-> Split created successfully!")

# Whisper MUST be 16000Hz
print("-> Casting audio streams to 16,000Hz (Whisper requirement)...")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\']'

def prepare_dataset(batch):
    # 1. Load Audio
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # 2. Normalize and Tokenize Text
    text = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    batch["labels"] = tokenizer(text).input_ids
    
    # 3. Calculate length to filter out > 30s audio
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch

# Filter length to avoid OOM
def is_audio_in_length_range(length):
    return length < MAX_DURATION_IN_SECONDS

print("-> Applying mapping function to extract features and label text. (This may take several minutes)...")
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
print("-> Mapping complete! Now filtering out audio longer than 30 seconds...")

common_voice = common_voice.filter(is_audio_in_length_range, input_columns=["input_length"])
print("-> Dataset preprocessing finished successfully.")

# ==========================================
# 3. METRICS (WER Calculation)
# ==========================================
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
print(f"Downloading/Loading pre-trained Whisper model: '{MODEL_NAME}'...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
print("-> Model loaded into memory successfully.")

# Disable forced tokens to allow it to speak Marathi naturally over training
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False # Needed for gradient checkpointing

# ==========================================
# 5. DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels to max length
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Strip decoder start token id from labels if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ==========================================
# 6. LOCAL METRICS LOGGER
# ==========================================
class LocalMetricsLogger(TrainerCallback):
    """
    Saves the training state log history (loss, eval, wer, epochs) 
    to a local JSON file every time logging occurs.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "metrics_history.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # We save the entire history of metrics the Trainer has collected
        with open(self.log_file, "w") as f:
            json.dump(state.log_history, f, indent=4)
        print(f"Metrics saved to {self.log_file}")

# ==========================================
# 7. TRAINING CONFIGURATION
# ==========================================
print("Configuring the Seq2Seq Trainer based on config.yaml...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    warmup_steps=config["training"]["warmup_steps"],
    max_steps=config["training"]["max_steps"],
    gradient_checkpointing=config["training"]["gradient_checkpointing"],
    bf16=config["training"]["bf16"], 
    evaluation_strategy="steps",
    predict_with_generate=True, # crucial for WER logging
    generation_max_length=config["training"]["generation_max_length"],
    save_steps=config["training"]["save_steps"],
    eval_steps=config["training"]["eval_steps"],
    logging_steps=config["training"]["logging_steps"],
    report_to=["wandb"], # Automates rich visualizations of GPU, Loss, and WER
    run_name=f"whisper-small-marathi-rtx6000", # Easier Wandb identification
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True, # Uploads to HuggingFace
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"], 
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[LocalMetricsLogger(OUTPUT_DIR)]
)

# ==========================================
# 8. EXECUTION (Train & Evaluate)
# ==========================================
if __name__ == "__main__":
    print("Checking for existing checkpoints to resume training...")
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else None

    if last_checkpoint is not None:
        print(f"Found checkpoint at {last_checkpoint}. Will resume training from there.")
    else:
        print("No checkpoint found. Starting training from scratch.")
        print("Evaluating BEFORE training to log baseline WER...")
        initial_metrics = trainer.evaluate()
        print(f"Zero-shot WER: {initial_metrics['eval_wer']}%")

    print("Starting Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("Evaluating AFTER training...")
    final_metrics = trainer.evaluate()
    print(f"Finetuned WER: {final_metrics['eval_wer']}%")

    print("Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()
    print("Done!")
