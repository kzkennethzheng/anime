import logging
from typing import Iterable

from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftMixedModel
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import anilist.puller

# Load model and tokenizer
MODEL_NAME = "tiiuae/falcon-rw-1b"

logger = logging.getLogger(__name__)
DEVICE = torch.device("cpu")


# Apply LoRA
def configure_model(model) -> PeftModel | PeftMixedModel:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],  # Falcon-specific
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def format_data(media: dict[str, any], review: dict[str, any]) -> str:
    return f"""title: {media["title"]}
        description: {media["description"]}
        genres: {media["genres"]}
        review: {review["body"]}
    """


def make_data(data: Iterable[tuple[dict[str, any], dict[str, any]]]) -> Iterable[str]:
    for media, review in data:
        yield format_data(media, review)


class TextDataset(IterableDataset):
    def __init__(self, data: Iterable[str], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for text in self.data:
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }


def fine_tune(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> None:
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.info("[Epoch %d, Step %d] Loss: %.4f", epoch, step, loss.item())


def main() -> None:
    logger.info("Retrieving model %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = configure_model(model)

    logger.info("Pulling data from anilist")
    data = anilist.puller.get_data()
    logger.info("Formatting and tokenizing data")
    dataset = TextDataset(make_data(data), tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    logger.info("Configuring model trainer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.to(DEVICE)
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Fine tuning model")
    fine_tune(model, dataloader, optimizer)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(name)s:%(levelname)s:%(message)s")
    main()

"""
from transformers import pipeline
pipe = pipeline("text-generation", model="./output", tokenizer=tokenizer)
pipe("Title: Breaking Bad\nDescription: ...\nPrompt: Write a review...")
"""
