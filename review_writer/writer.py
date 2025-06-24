import logging
from typing import Callable, Iterable

from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftMixedModel
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import anilist.puller

MODEL_NAME = "tiiuae/falcon-rw-1b"
MODEL_FINE_TUNED_SAVE_FOLDER = "falcon-lora-review-finetuned"
MODEL_ORIGINAL_SAVE_FOLDER = "falcon-lora-review"

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
N_EPOCHS = 2
LEARNING_RATE = 5e-5


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


def format_media_review(media: dict[str, any], review: dict[str, any]) -> str:
    return f"""title: {media["title"]}
        description: {media["description"]}
        genres: {media["genres"]}
        Write a review of {media["title"]}...
        review: {review["body"]}
    """


def format_data(data: Iterable[tuple[dict[str, any], dict[str, any]]]) -> Iterable[str]:
    for media, review in data:
        yield format_media_review(media, review)


def make_data() -> Iterable[str]:
    data = anilist.puller.get_data()
    return format_data(data)

class TextDataset(IterableDataset):
    def __init__(self, data_initializer: Callable[..., Iterable[str]], tokenizer, max_length=512):
        self.data_initializer = data_initializer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for text in self.data_initializer():
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
    for epoch in range(N_EPOCHS):
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
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(name)s:%(levelname)s:%(message)s")

    logger.info("Retrieving model %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = configure_model(model)

    logger.info("Pulling data from anilist")
    data = anilist.puller.get_data()
    logger.info("Formatting and tokenizing data")
    dataset = TextDataset(make_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    logger.info("Configuring model trainer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.to(DEVICE)
    model.train()
    tokenizer.pad_token = tokenizer.eos_token

    model.save_pretrained(MODEL_ORIGINAL_SAVE_FOLDER)
    tokenizer.save_pretrained(MODEL_ORIGINAL_SAVE_FOLDER)

    logger.info("Fine tuning model")
    fine_tune(model, dataloader, optimizer)

    model.save_pretrained(MODEL_FINE_TUNED_SAVE_FOLDER)
    tokenizer.save_pretrained(MODEL_FINE_TUNED_SAVE_FOLDER)


if __name__ == "__main__":
    main()
