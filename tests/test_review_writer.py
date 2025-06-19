from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.onnx import main_export
from peft import PeftModel
import torch
from transformers import pipeline, AutoModelForCausalLM
from transformers.pipelines.text_generation import TextGenerationPipeline

import review_writer.writer as writer

FINETUNED_MODEL_FOLDER = "./" + writer.MODEL_FINE_TUNED_SAVE_FOLDER
FINETUNED_TOKENIZER_FOLDER = "./" + writer.MODEL_FINE_TUNED_SAVE_FOLDER
MERGED_MODEL_FOLDER = "falcon-merged-review-finetuned"
ORIGINAL_MODEL_FOLDER = "./" + writer.MODEL_ORIGINAL_SAVE_FOLDER
ORIGINAL_TOKENIZER_FOLDER = "./" + writer.MODEL_ORIGINAL_SAVE_FOLDER

ONNX_MODEL_FOLDER = "falcon-merged-review-finetuned-onxx"

# model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_FOLDER)
# model = torch.compile(model)
model = ORTModelForCausalLM.from_pretrained(MERGED_MODEL_FOLDER, export=True)
finetuned_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=FINETUNED_TOKENIZER_FOLDER,
)
original_pipe = pipeline(
    "text-generation", model=ORIGINAL_MODEL_FOLDER, tokenizer=ORIGINAL_TOKENIZER_FOLDER
)

main_export(
    model_name_or_path=MERGED_MODEL_FOLDER,
    output=ONNX_MODEL_FOLDER,
    task="text-generation",
)


test_media = [
    {
        "title": "Ave Mujica - The Die is Cast",
        "description": """
"Will you give me the rest of your life?"

After losing everything in a single night, Sakiko Togawa reaches out toward an even deeper abyss, one that will drag everyone around her down as well. Gathering the lives of girls burdened with their own troubles and desires, Sakiko raises the curtain on a perfect masquerade.

"Welcome to the world of Ave Mujica."

On a stage where sorrow, death, fear, love—even the solace of forgetting—are stripped away, will their masks be torn off and shatter into oblivion, or will they….
""",
        "genres": "['Band', 'Female Protagonist', 'Primary Female Cast', 'Primary Teen Cast', 'Acting', 'Full', 'CGI', 'Yuri', 'Metal Music', 'Ensemble Cast']",
    },
    {
        "title": "To Be Hero X",
        "description": """
This is a world where heroes are created by people's trust, and the hero who gains the most trust is known as X. 
In this world, people's trust can be quantified through data, and these values are reflected on everyone's wrist. 
As long as one gains enough trust points, an ordinary person can possess superpowers and become a superhero who saves the world. 
However, the constantly changing trust values make the path of a hero full of uncertainties... 
""",
        "genres": "['Urban Fantasy', 'Conspiracy', 'Superhero', 'Henshin', 'Anti-Hero', 'Idol', 'Super Power', 'CGI', 'Anthropomorphism', 'Mixed Media']",
    },
]

prompts = [
    "Write a review of ",
    "Write a positive review of ",
    "Write a negative review of ",
    "Write a glowing review of ",
    "Write a bittersweet review of ",
    "Write a long review of ",
    "Write an elitist review full of gatekeeping.",
]


def merge_models():
    base_model = AutoModelForCausalLM.from_pretrained(writer.MODEL_NAME)
    peft_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_FOLDER)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(MERGED_MODEL_FOLDER)


def format_media(media: dict[str, any], prompt: str) -> str:
    if prompt[-1] == " ":
        prompt = prompt + media["title"]
    return f"""title: {media["title"]}
        description: {media["description"]}
        genres: {media["genres"]}
        {prompt}...
    """


def apply_pipeline(
    generator: TextGenerationPipeline,
    media: dict[str, any],
    max_new_tokens=50,
    prompt=prompts[0],
):
    output = generator(
        format_media(media, prompt),
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )
    return output[0]["generated_text"]


# def test_pipelines():
#     print(apply_pipeline(finetuned_pipe, test_media[0]))
#     print(
#         "================================================================================"
#     )
#     print(apply_pipeline(original_pipe, test_media[0]))
#     assert 0 == 0


if __name__ == "__main__":
    # merge_models()
    print(
        apply_pipeline(
            finetuned_pipe, test_media[0], max_new_tokens=50, prompt=prompts[-1]
        )
    )
    # print(
    #     "================================================================================"
    # )
    # print(apply_pipeline(original_pipe, test_media[0]))
