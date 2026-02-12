from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import os
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cpu"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)

        # Load transformed dataset
        dataset = load_from_disk(self.config.data_path)

        # 🔥 Reduce dataset size for CPU stability
        train_dataset = dataset["train"].select(range(200))
        eval_dataset = dataset["validation"].select(range(50))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )

        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,

            # 🔥 Transformers 5.x change
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            report_to="none",
            dataloader_num_workers=0
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        trainer.train()

        # Save model
        model.save_pretrained(
            os.path.join(self.config.root_dir, "t5-small-model")
        )

        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )
