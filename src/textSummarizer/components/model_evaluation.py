from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import evaluate
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig
import os

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="dialogue",
        column_summary="summary",
    ):

        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )

        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches)
        ):

            # T5 requires prefix
            article_batch = ["summarize: " + article for article in article_batch]

            inputs = tokenizer(
                article_batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                num_beams=4,
                length_penalty=1.0,
                max_length=128,
            )

            decoded_summaries = [
                tokenizer.decode(
                    s,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for s in summaries
            ]

            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch
            )

        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load fine-tuned T5 model (LOCAL)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path,
            local_files_only=True
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path,
            local_files_only=True
        ).to(device)

        # Load dataset
        dataset = load_from_disk(self.config.data_path)

        # Load ROUGE metric
        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset["test"][0:10],
            rouge_metric,
            model,
            tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )

        # Ensure directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Save results
        df = pd.DataFrame([score])
        df.to_csv(self.config.metric_file_name, index=False)

        print("Evaluation Completed")
        print(score)