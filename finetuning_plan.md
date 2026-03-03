# Whisper Finetuning Plan for Marathi ASR

**Hardware Context:** NVIDIA RTX Pro 6000 (Blackwell architecture) with 96GB VRAM.
**Model:** `openai/whisper-small` (244M variables).
**Dataset:** `MatrixSpeechAI/All_Marathi_ASR`.

## 1. Overall Strategy & Hardware Optimization
Your RTX Pro 6000 has 96GB of VRAM and fully supports **BF16 (Bfloat16) mixed-precision**. Because Whisper-small is relatively lightweight for this GPU, **full fine-tuning** (updating all weights) is easily achievable and will yield significantly better results than PEFT/LoRA. You can use large batch sizes combined with gradient accumulation to accelerate training while keeping memory in check.

## 2. Step-by-Step Execution Plan

### Step 2.1: Setup and Authentication
You need the absolute latest libraries for optimal BF16 and Whisper integration.
```bash
pip install -q transformers datasets evaluate jiwer accelerate wandb soundfile librosa
```
Authenticate with HuggingFace (to push the model) and Weights & Biases (for metrics):
```bash
huggingface-cli login
wandb login
```

### Step 2.2: Data Preparation
1. **Load Dataset**: Pull `MatrixSpeechAI/All_Marathi_ASR`.
2. **Audio Resampling**: Whisper strict requirement: **16,000 Hz**. `datasets` provides a convenient `.cast_column("audio", Audio(sampling_rate=16000))` utility for this.
3. **Filtering**: Whisper can only process 30-second audio intervals. *You must filter out any audio longer than 30 seconds to prevent OOM errors and hallucinations.*
4. **Text Normalization**: Strip punctuation and standardising spacing (especially important for Marathi characters) so string variations don't artificially inflate your Word Error Rate (WER).

### Step 2.3: Architecture Configuration
- **Feature Extractor**: Processes audio array into Log-Mel Spectrograms.
- **Tokenizer**: Converts text to token IDs. You *must* specify `language="Marathi"` and `task="transcribe"`.
- **Processor**: Combines the feature extractor and tokenizer.

### Step 2.4: Measuring "Before" WER (Zero-Shot Evaluation)
To measure how the model performs *before* training:
1. Load the pre-trained `WhisperForConditionalGeneration`.
2. Run inference on a test split of your dataset using `.generate()`.
3. Compute the WER between ground truth and predictions using the `evaluate` library.

### Step 2.5: Training Configuration (`Seq2SeqTrainer`)
- **Mixed Precision**: Use `bf16=True`. It is more stable than `fp16` on Blackwell architectures.
- **Gradient Checkpointing**: Use `gradient_checkpointing=True` to save VRAM, though 96GB might fit it even without.
- **Predict with Generate**: Crucial setting: `predict_with_generate=True` in your training arguments. Models must predict autoregressively to compute genuine WER during evaluation steps.
- **Generation Settings**: Set `model.config.forced_decoder_ids` to force the model to output Marathi.
- **Suppress Tokens**: `model.config.suppress_tokens = []` allows the model to predict without arbitrary OpenAI constraints.

### Step 2.6: Visualization (WandB)
- In the `TrainingArguments`, set `report_to="wandb"`. 
- This automatically logs Train Loss, Validation Loss, WER, and Steps. You will see live graphical curves in your WandB dashboard during training.

### Step 2.7: Measuring "After" WER & Uploading
- The `Seq2SeqTrainer` automatically evaluates the final model, providing the definitive "Before vs. After" WER comparison metrics if you evaluated at Step 2.4.
- Because `push_to_hub=True` is provided in `TrainingArguments`, the trainer will upload precision metrics, model `.safetensors` files, the processor, and a `README.md` containing training graphs to your HuggingFace repository immediately after completion. 

---

## 3. Critical Things Commonly Missed (Best Practices)

1. **Language and Task IDs**: 
   If you don't explicitly pass `<|mr|>` and `<|transcribe|>` decoder tokens during fine-tuning, the model might try to translate instead of transcribe, or predict English characters.
2. **Text Normalization (WER killer)**: 
   If your dataset transcribes "मराठी." and the model predicts "मराठी", the default WER calculator treats this as an error. For Indian languages, utilizing `jiwer` with robust regex text-stripping is critical.
3. **Max Input Length**: 
   Whisper crashes on > 30s. Filter your dataset map with a condition checking length `< 30.0` seconds or `480000` samples.
4. **Learning Rate**: 
   Whisper finetuning is highly sensitive to the learning rate. Usually, `1e-5` with a `linear` warmup scheduler (e.g., 500 warmup steps) yields the most stable convergence without catastrophic forgetting.
