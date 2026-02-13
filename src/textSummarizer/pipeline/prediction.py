import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer (local)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path,
            local_files_only=True
        )

        # Load trained model (local)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path,
            local_files_only=True
        ).to(self.device)

    def predict(self, text: str):

        # T5 requires prefix
        input_text = "summarize: " + text

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=128,
            length_penalty=1.0
        )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary
