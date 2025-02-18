from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
from tqdm import tqdm
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="mscoco")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()
    return args



def context_prompt(caption):
    template_caption=f"""You are a text rewriting expert. Given a description text: {caption}. Please rephrase this provided description text into 3 different expressions, with each no longer than 15 words. Return the results in the JSON format as follows: {{"1": "text1", "2": "text2", "3": "text3"}}."""
    return template_caption

def match_json_res(response):
    pattern = r'\{\s*"1"\s*:\s*"(.*?)",\s*"2"\s*:\s*"(.*?)",\s*"3"\s*:\s*"(.*?)"\s*\}'
    matches = re.findall(pattern, response)
    return matches


def llama_repharsing(args):
    rephrasing_result={}
    res_path = f"../data/{args.dataset}/train/llm_rephrasing_cap.json"    
    train_captions=json.load(open(f"../data/{args.dataset}/train_captions.json"))

    model_id = "facebook/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        device_map=args.device,
    )
    model.eval()


    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    data=list(set(train_captions))
    batch=args.batch_size
    for i in tqdm(range(int(len(data)/batch)+1)):
        
        batch_data=data[i*batch:(i+1)*batch]
        batch_data_prompted=[context_prompt(item) for item in batch_data]

        input_ids = tokenizer(
            batch_data_prompted,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            **input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        ouput_text=tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt,cap,response in zip(batch_data_prompted,batch_data,ouput_text):
            texts_from_rephrasing=match_json_res(response.replace(prompt,""))
            if texts_from_rephrasing!=[]:
                rephrasing_result[cap]=list(texts_from_rephrasing[0])
        
        if i%500==0:
            with open(res_path,"w") as f:
                json.dump(rephrasing_result,f)
        # break

    with open(res_path,"w") as f:
        json.dump(rephrasing_result,f)


if __name__ == "__main__":
    args = get_args()
    llama_repharsing(args)
