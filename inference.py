import sys
sys.path.insert(0,"../SaCap")
import clip
import torch
from model.captioner_model import Captioner
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
import tqdm
import argparse
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.clipscore import *


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def feature_project(image_global_feature,support_feature):
    with torch.no_grad():
        image_global_feature/=image_global_feature.norm(dim=-1,keepdim=True)
        cosine_similarity = image_global_feature @ support_feature.T.float()
        cosine_similarity = (cosine_similarity / args.temperature).softmax(dim=-1)
        project_embedding = cosine_similarity @ support_feature.float()
        project_embedding /= project_embedding.norm(
            dim=-1, keepdim=True)
    return project_embedding.detach()


class TestDataset(Dataset):
    def __init__(self,preprocess, image_feature_file, support_text_feature_file, image_root, retrieval_hard_prompt_file, tokenizer):
        self.tokenizer = tokenizer
        self.image_root=image_root
        self.preprocess=preprocess
        self.support_feature_dict = torch.load(support_text_feature_file, map_location="cpu")
        self.support_feature = torch.vstack(list(self.support_feature_dict.values()))
        self.support_feature_norm = (self.support_feature/self.support_feature.norm(dim=-1, keepdim=True))

        self.image_global_feature_dict = torch.load(
            image_feature_file, map_location="cpu")
        self.image_feature = torch.vstack(
            list(self.image_global_feature_dict.values()))
        self.img_names = list(self.image_global_feature_dict.keys())

        self.hard_prompt_dict = json.load(open(retrieval_hard_prompt_file))
        self.hard_prompt_dict = {i["pk"]: i["hard_prompt"][:args.retrieve_object_phrase_num] for i in self.hard_prompt_dict}

        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image_name = self.img_names[index]
        image_global_feature = self.image_global_feature_dict[image_name]

        image_path = self.image_root+f"/{image_name}.jpg"
        image_instance = Image.open(image_path)
        image_instance_input = self.preprocess(image_instance)
        image_instance.close()

        hard_prompt_list = self.hard_prompt_dict[image_name]
        hard_prompt = ",".join(hard_prompt_list)
        input_token, attention_mask = self.make_input(hard_prompt)
        return image_name, image_global_feature, input_token, attention_mask, image_instance_input

    def make_input(self, hard_prompt):
        tokenizer = self.tokenizer
        bos_token = tokenizer.encode(":")[0]

        hard_prompt_input = tokenizer(hard_prompt, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=args.max_prompt_length)

        pad_id = tokenizer.pad_token_id
        soft_prompt_token = torch.full((1, args.prefix_length), pad_id)
        soft_prompt_mask = torch.ones_like(soft_prompt_token)

        bos_token = torch.full((1, 1), bos_token)
        bos_mask = torch.ones_like(bos_token)

        input_token = torch.concat(
            [soft_prompt_token, hard_prompt_input["input_ids"], bos_token], dim=-1)
        attention_mask = torch.concat(
            [soft_prompt_mask, hard_prompt_input["attention_mask"], bos_mask], dim=-1)
        return input_token, attention_mask


def collate(batch):
    image_name, image_global_feature, input_token, attention_mask, image_instance_input = zip(
        *batch)
    image_global_feature = torch.vstack(image_global_feature)
    input_token = torch.vstack(input_token)
    attention_mask = torch.vstack(attention_mask)
    image_instance_input=torch.stack(image_instance_input)
    return image_name, image_global_feature, input_token, attention_mask, image_instance_input

def save_args_to_file(filename, args):
    with open(filename, 'w') as file:
        json.dump(args.__dict__, file, indent=4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--test_dataset", type=str, default="mscoco")
    parser.add_argument("--support_dataset", type=str, default="mscoco")
    parser.add_argument("--model_path", type=str, default="../trained_model/mscoco")
    parser.add_argument("--prefix_length", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=20)
    parser.add_argument("--max_generate_length", type=int, default=30)
    parser.add_argument("--retrieve_object_phrase_num", type=int, default=3)
    args = parser.parse_args()
    return args


def main(args):

    test_data_out_dir=os.path.join(args.model_path,f'test_in_{args.test_dataset}')
    print("out_dir: ",test_data_out_dir)
    if not os.path.exists(test_data_out_dir):
        os.makedirs(test_data_out_dir)

    set_seed(args.seed)
    device = args.device
    captioner = Captioner(prefix_len=args.prefix_length)
    

    clip_model,image_preprocess=clip.load("ViT-B/32", jit=False)
    image_model = clip_model.visual
    clip_model.to(device)
    image_model.forward = image_model.grid_feature

    file_root=f"./data/{args.test_dataset}/test"
    references_file=f"./data/{args.test_dataset}/refs.json"

    if args.test_dataset=="nocaps":
        test_data_out_dir=os.path.join(args.model_path,f'test_in_{args.test_dataset}/overall')
        print("out_dir: ",test_data_out_dir)
        if not os.path.exists(test_data_out_dir):
            os.makedirs(test_data_out_dir)
        
        references_file_dict={"in_domain":f"./data/{args.test_dataset}/test/refs_indomain.json",
                              "near_domain":f"./data/{args.test_dataset}/test/refs_neardomain.json",
                              "out_domain":f"./data/{args.test_dataset}/test/refs_outdomain.json"}
    
        test_file_out_root={"in_domain":os.path.join(args.model_path,f'test_in_{args.test_dataset}/in_domain'),
                              "near_domain":os.path.join(args.model_path,f'test_in_{args.test_dataset}/near_domain'),
                              "out_domain":os.path.join(args.model_path,f'test_in_{args.test_dataset}/out_domain')}
        for nocap_domain_out_dir in test_file_out_root.values():
            if not os.path.exists(nocap_domain_out_dir):
                os.makedirs(nocap_domain_out_dir)

        test_img_domain_map=json.load(open(f"./data/{args.test_dataset}/test/img_domain_map.json"))

    test_data = TestDataset(retrieval_hard_prompt_file=file_root+"/test_image_hard_prompt_dict.json",
                           image_feature_file=file_root+"/test_image_global_feature_dict.pt",
                           image_root=file_root+"/test_images",
                           support_text_feature_file=f"./data/{args.support_dataset}/train/text_feature_dict.pt", 
                           tokenizer=captioner.tokenizer,
                           preprocess=image_preprocess)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=128, 
                                 num_workers=8, 
                                 shuffle=False, 
                                 drop_last=False, 
                                 collate_fn=collate)
    
    if os.path.isfile(args.model_path):
        captioner = Captioner(prefix_len=args.prefix_length)
        captioner.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        captioner.to(device)
        captioner.eval()

        set_seed(args.seed)
        res_dict = {}
        for image_name, image_global_feature, input_token, attention_mask,image_instance_input in tqdm.tqdm(test_dataloader):
            image_global_feature = image_global_feature.to(device)
            input_token = input_token.to(device)
            attention_mask = attention_mask.to(device)
            image_instance_input = image_instance_input.to(device)
        
            image_pj_feature=feature_project(image_global_feature,test_data.support_feature_norm.to(device)).float()
            soft_prompt = captioner.map_layer(image_pj_feature)
            image_grid_feature = clip_model.encode_image(image_instance_input).float()
            
            
            batch_res_dict=captioner.batch_caption_generation(image_pk=image_name,
                                                    input_token=input_token,
                                                    attention_mask=attention_mask,
                                                    max_generate_length=args.max_generate_length,
                                                    soft_prompt=soft_prompt,
                                                    image_grid_feature=image_grid_feature)
            res_dict.update(batch_res_dict)
            
        res_save_path = args.model_path.replace(".pt","_res.json")
        with open(res_save_path, 'w') as f:
            json.dump(res_dict, f)
        eval_metrics(res_save_path,references_file)

    elif os.path.isdir(args.model_path):
        
        for root, dirs, files in os.walk(args.model_path):
            for file in files:
                if file.endswith('.pt'):
                    model_path=os.path.join(root, file)
                    captioner.load_state_dict(torch.load(model_path, map_location="cpu"))
                    captioner.to(device)
                    captioner.eval()
                    set_seed(args.seed)

                    res_dict = {}
                    for image_name, image_global_feature, input_token, attention_mask,image_instance_input in tqdm.tqdm(test_dataloader):
                        image_global_feature = image_global_feature.to(device)
                        input_token = input_token.to(device)
                        attention_mask = attention_mask.to(device)
                        image_instance_input = image_instance_input.to(device)
                    
                        image_pj_feature=feature_project(image_global_feature,test_data.support_feature_norm.to(device)).float()
                        soft_prompt = captioner.map_layer(image_pj_feature)
                        image_grid_feature = clip_model.encode_image(image_instance_input).float()
                        
                        batch_res_dict=captioner.batch_caption_generation(image_pk=image_name,
                                                                input_token=input_token,
                                                                attention_mask=attention_mask,
                                                                max_generate_length=args.max_generate_length,
                                                                soft_prompt=soft_prompt,
                                                                image_grid_feature=image_grid_feature)
                        res_dict.update(batch_res_dict)
                    res_save_path = os.path.join(test_data_out_dir,file.replace(".pt","_res.json"))
                    with open(res_save_path, 'w') as f:
                        json.dump(res_dict, f)
                    eval_metrics(res_save_path,references_file)

                    if args.test_dataset=="nocaps":
                        doamin_cap={"in_domain":{},
                                    "near_domain":{},
                                    "out_domain":{}}
                        for k,v in test_img_domain_map.items():
                            doamin_cap[v][k]=res_dict[k]
                        for domain in doamin_cap:
                            domain_res_save_path = os.path.join(test_file_out_root[domain],file.replace(".pt","_res.json"))
                            with open(domain_res_save_path, 'w') as f:
                                json.dump(doamin_cap[domain], f)
                            eval_metrics(domain_res_save_path,references_file_dict[domain])
                            
        
        save_args_to_file(os.path.join(test_data_out_dir,"test_args.json"),args)

        output_filename =os.path.join(test_data_out_dir,f'all_res.txt')
        file_list = []
        for filename in os.listdir(test_data_out_dir):
            if filename.endswith('.res'):
                file_list.append(filename)

        file_list.sort(key=lambda x: int(x.split('_')[1]))

        with open(output_filename, 'w') as output_file:
            for filename in file_list:
                file_path = os.path.join(test_data_out_dir, filename)
                with open(file_path, 'r') as input_file:
                    output_file.write(f'File: {filename}\n')  
                    output_file.write(input_file.read())
                    output_file.write('\n')  


        if args.test_dataset=="nocaps":
            for domain in test_file_out_root:
                output_filename =os.path.join(test_file_out_root[domain],f'all_res.txt')
                file_list = []
                for filename in os.listdir(test_file_out_root[domain]):
                    if filename.endswith('.res'):
                        file_list.append(filename)

                file_list.sort(key=lambda x: int(x.split('_')[1]))

                with open(output_filename, 'w') as output_file:
                    for filename in file_list:
                        file_path = os.path.join(test_file_out_root[domain], filename)
                        with open(file_path, 'r') as input_file:
                            output_file.write(f'File: {filename}\n')  
                            output_file.write(input_file.read())
                            output_file.write('\n')  

if __name__ == "__main__":
    args = get_args()
    main(args)
