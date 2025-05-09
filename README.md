# FineTuning-using-Transformers

# uv Environment Setup

### Download Astral UV
**windows**:
>powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

### So For Each Task Go into Respective Tasks Directory 
### And Then Type Command (So it downloads all required Dependencies) for that Fine Tuning Task:
>uv sync

### for running any python file:
>uv run filename

**Example (I am inside bert-FineTuning Directory)**

```uv run LoraFineTuning.py```

# Task-1: 
### Fine Tuning bert-based-cased Model on imdb dataset for sentiment Classification Task

#### Here First Download and save imdb dataset in bert-FineTuning Directory. And Then you can Run Any Fine Tuning File, As in Code Files its Taking imdb Dataset from this Folder which can't be uploaded due to size Constraints.Then Run File using uv as explained Above

### downloading and saving dataset:
```
from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("imdb")

# Save it to the current directory
dataset.save_to_disk("./imdb_dataset")

print("IMDb dataset saved to ./imdb_dataset")
```

***StandardFineTuning.py**: Fine-Tuning Without any technique like lora,Quatization, Adaptive Lora

```uv run StandardFineTuning.py```

**LoraFineTuning.py**: Fine Tuning Using Lora technique

```uv run LoraFineTuning.py```

### Below are Hyperparamter Tuning for Lora Parameters with respect to LorFinetuning File parameters

**LoraFineTuning_1_HalvedRank_IncreasedLR.py**: Halved rank and Increased Learning rate

```uv run LoraFineTuning_1_HalvedRank_IncreasedLR.py```

**LoraFineTuning_2_DoubledRank_HalvedLR.py**: Doubled rank and Decreased Learning rate'

```uv run LoraFineTuning_2_DoubledRank_HalvedLR.py```

**LoraFineTuning_3_HalvedLora_alpha_IncreaseEpochs.py**: Halved Lora Alpha and Increased Epochs

```uv run LoraFineTuning_3_HalvedLora_alpha_IncreaseEpochs.py```

# Task-2: 
### Fine Tuning t5 Model on Code Search Net Dataset for Code Generation Task

#### Here Enter in t5_finetune Directory
**main.py**: Downloads the CodeSearchNet Python dataset, preprocesses & tokenizes it, then runs full-scale fine-tuning of T5 via Standard, LoRA, and Adapter methods.

```uv run main.py```

**preprocess.py**: Loads a 10% slice of CodeSearchNet, preprocesses and tokenizes it, and offers rapid small-scale fine-tuning (SFT, LoRA, Adapter) with BLEU evaluation.

```uv run preprocess.py```

**evaluate2.py**: Loads the preprocessed/test data and each fine-tuned model to compute BLEU, ROUGE, syntax correctness, generation time, and saves both CSVs and comparative plots.

```uv run evaluate2.py```

**query.py**: Provides an interactive inference pipeline that loads each model, generates code for user samples, measures BLEU/ROUGE/syntax, and writes per-sample & summary CSVs.

```uv run query.py```


# Task 3:
### Fine Tuning Roberta-base model on CNN/DailyMail Dataset for Summarization Task 

#### Here after entering into RobertA-FineTuning

**First Run Summarization_setup.py (Stuff Like Loading model and dataset)**

```uv run Summarization_setup.py```

**Then Run FineTuning_Summarization (Training)**

```uv run FineTuning_Summarization.py```

**If want to Compare Summaries generated by before and after Finetuning Model**

```uv run Summarization_Comparison.py```

**When we ran our Comparison log file is present**: Summarization_Comparison.out


