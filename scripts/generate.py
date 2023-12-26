from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import os, json
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--template", type=str, default="../templates/alpaca.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=bool, default=False)


    return parser.parse_args()


def evaluate(prompt, model, tokenizer, spliter, device='cuda'):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generation_output = model.generate(input_ids=input_ids, num_beams=1, num_return_sequences=1,
                                        max_new_tokens=2048, temperature=0., top_p=1)
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    output = output.split(spliter, 1)[-1]
    return output


def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = {'device_map': 'auto'}
    else:
        device_arg = {'device_map': {"": args.device}}

    print(f"Loading base model: {args.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path,
                                                 return_dict=True,
                                                 torch_dtype=torch.float16,
                                                 **device_arg)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    generator = args.base_model_name_or_path.split("/")[-1]
    with open(args.template) as fp:
        template = json.load(fp)

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval",
                                     "alpaca_eval")["eval"]
    save_data = []
    for i_e, example in tqdm(enumerate(eval_set), desc='generating'):

        instruction = example["instruction"]
        input_text = template["prompt_no_input"].format(
            instruction=instruction)
        spliter = template['response_split']
        if "dpo" in args.base_model_name_or_path:
            input_text = template["prompt_dpo"].format(instruction=instruction)
            spliter = template['dpo_split']
        output = evaluate(input_text, model, tokenizer, spliter)

        if args.verbose:
            print("instruction", instruction)
            print("output", output)
            input()
        save_one = {
            "instruction": example["instruction"],
            "output": output.strip(),
            "generator": generator,
            'dataset': example['dataset'],
        }
        save_data.append(save_one)

    save_file = os.path.join(args.base_model_name_or_path, "alpaca_eval.json")
    with open(save_file, 'w') as writer:
        json.dump(save_data, writer, ensure_ascii=False, indent=2)


if __name__ == "__main__" :
    main()
