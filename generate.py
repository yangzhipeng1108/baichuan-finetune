import os
import sys

import argparse
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from prompter import Prompter
from model import MODE

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    prompt_template: str = ""
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(base_model)
    if device == "cuda":
        model = MODE[args.mode]["model"].from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif device == "mps":
        model = MODE[args.mode]["model"].from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    else:
        model = MODE[args.mode]["model"].from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    while True:
        instruction = input("Input:")
        if len(instruction.strip()) == 0:
            break
        print("Response:", evaluate(instruction))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    parser.add_argument("--mode", type=str, default="glm2", help="")
    args = parser.parse_args()
    main(args.load_8bit, args.base_model, "")
