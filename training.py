import sys
sys.path.insert(0,"../SaCap")
import clip
from model.captioner_model import Captioner
import torch
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from tqdm import tqdm
import argparse
import os
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import datetime

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

class TrainDataset(Dataset):
    def __init__(self,
                 preprocess, 
                 image_feature_file,
                 synthetic_image_hard_prompt_file,
                 synthetic_image_pk_file, 
                 synthetic_image_root, 
                 text_feature_file, 
                 tokenizer, 
                 supporting_image_hard_prompt_file, 
                 supporting_image_feature_file, 
                 rephrasing_num_cap_file, 
                 repharsing_image_root, 
                 cap_rephrasing_cap_file,
                 ):
        self.tokenizer = tokenizer
        self.preprocess=preprocess
        self.cap_rephrasing_cap_file=json.load(open(cap_rephrasing_cap_file))
        self.captions=list(self.cap_rephrasing_cap_file.keys())

        pk2cap_dict = json.load(open(synthetic_image_pk_file))
    
        self.cap2imgPath={cap:os.path.join(synthetic_image_root,f"{pk}.jpg") for pk,cap in pk2cap_dict.items()}
        
        rephrasing_num_cap_dict=json.load(open(rephrasing_num_cap_file))
        for num in rephrasing_num_cap_dict:
            self.cap2imgPath[rephrasing_num_cap_dict[num]]=os.path.join(repharsing_image_root,f"{num}.jpg")


        self.hard_prompt_dict = json.load(open(synthetic_image_hard_prompt_file))
        self.hard_prompt_dict = {item["pk"]: item["hard_prompt"][:args.retrieve_object_phrase_num_q] for item in self.hard_prompt_dict}

        repharsing_image_hard_prompt_dict=json.load(open(supporting_image_hard_prompt_file))
        self.hard_prompt_dict.update({item["pk"]: item["hard_prompt"][:args.retrieve_object_phrase_num_q] for item in repharsing_image_hard_prompt_dict})

        self.image_glocal_feature_dict=torch.load(image_feature_file,map_location="cpu")
        self.image_glocal_feature_dict.update(torch.load(supporting_image_feature_file,map_location="cpu"))
        self.text_feature_dict = torch.load(text_feature_file,map_location="cpu")
        support_feature = torch.vstack(list(self.text_feature_dict.values()))
        self.support_feature_norm = (
            support_feature/support_feature.norm(dim=-1, keepdim=True))
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption=self.captions[index]
        image_pk=self.repairing_mechanism(caption)
        image_global_feature=self.image_glocal_feature_dict[image_pk]
        path = self.cap2imgPath[image_pk]

        # construct hard prompt
        hard_prompt_list = self.hard_prompt_dict[caption][:args.retrieve_object_phrase_num_q]
        hard_prompt_list = [i.strip(".") for i in hard_prompt_list]
        hard_prompt = ",".join(hard_prompt_list)

        image_instance = Image.open(path)
        image_input = self.preprocess(image_instance)
        image_instance.close()
        caption = ":"+caption.strip().split(".")[0] + '.'
        input_token, attention_mask, token_for_loss = self.make_input(
            caption, hard_prompt)
        return image_global_feature, input_token, attention_mask, token_for_loss, image_input

    # re-pairing mechanism
    def repairing_mechanism(self,caption):
        image_pk=caption
        image_global_feature = self.image_glocal_feature_dict[image_pk]
        text_feature=self.text_feature_dict[caption]
        sim=(image_global_feature@text_feature).item()

        supporting_image_feaures=torch.vstack([self.image_glocal_feature_dict[re_cap] for re_cap in self.cap_rephrasing_cap_file[caption]])
        sim_list=(text_feature@supporting_image_feaures.T).tolist()
        threshold=max(sim_list)

        if sim<=threshold:
            candidates=[i for i,s in enumerate(sim_list+[sim]) if s==threshold]
            retrieve_image_idx = random.choice(candidates)
            retrieve_image_pk = (self.cap_rephrasing_cap_file[caption]+[caption])[retrieve_image_idx]
            return retrieve_image_pk
        else:
            return image_pk


    def make_input(self, caption, hard_prompt):
        tokenizer = self.tokenizer
        # tokenize hard prompt and caption 
        hard_prompt = tokenizer(hard_prompt, 
                              return_tensors="pt", 
                              padding="max_length",
                              truncation=True, 
                              max_length=args.max_prompt_length)
        caption_input = tokenizer(caption, 
                                  return_tensors="pt", 
                                  padding="max_length",
                                  truncation=True, 
                                  max_length=args.max_generate_length)

        # make prefix input_id and attention_mask
        pad_id = tokenizer.pad_token_id
        prefix_token = torch.full((1, args.prefix_length), pad_id)
        prefix_mask = torch.ones_like(prefix_token)
        hard_prompt_token_for_loss = torch.full(hard_prompt["input_ids"].shape, pad_id)

        # concat tensor
        input_token = torch.concat([prefix_token, hard_prompt["input_ids"], caption_input["input_ids"]], dim=-1)
        attention_mask = torch.concat([prefix_mask, hard_prompt["attention_mask"], caption_input["attention_mask"]], dim=-1)
        token_for_loss = torch.concat([prefix_token, hard_prompt_token_for_loss, caption_input["input_ids"]], dim=-1)
        return input_token, attention_mask, token_for_loss


def collate(batch):
    caption_clip_feature, input_token, attention_mask, token_for_loss, image_instances = zip(*batch)
    caption_clip_feature = torch.vstack(caption_clip_feature)
    input_token = torch.vstack(input_token)
    attention_mask = torch.vstack(attention_mask)
    token_for_loss = torch.vstack(token_for_loss)
    image_instances=torch.stack(image_instances)
    return caption_clip_feature, input_token, attention_mask, token_for_loss, image_instances


def save_args_to_file(filename, args):
    with open(filename, 'w') as file:
        json.dump(args.__dict__, file, indent=4)




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--save_per_model", type=bool,default=True)
    parser.add_argument("--dataset", type=str,default="mscoco")
    parser.add_argument("--output_dir", type=str, default="./trained_model/mscoco")
    parser.add_argument("--retrieve_object_phrase_num_q", type=int, default=3)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--max_prompt_length", type=int, default=20)
    parser.add_argument("--max_generate_length", type=int, default=30)
    args = parser.parse_args()
    return args



def main(args):
    set_seed(args.seed)
    device = args.device

    captioner = Captioner(prefix_len=args.prefix_length)
    captioner.to(device)
    clip_model,image_preprocess=clip.load("ViT-B/32", jit=False)
    clip_model.to(device)
    image_model = clip_model.visual
    image_model.forward = image_model.grid_feature

    file_root=f"./data/{args.dataset}/train"
    train_data = TrainDataset(image_feature_file=file_root+"/synthetic_image_global_feature_dict.pt",
                              text_feature_file=file_root+"/text_feature_dict.pt", 
                              synthetic_image_hard_prompt_file=file_root+"/synthetic_image_hard_prompt_dict.json",
                            
                              synthetic_image_pk_file=file_root+"/synthetic_image_pk_dict.json",
                              
                              rephrasing_num_cap_file=file_root+"/supporting_image_pk_dict.json",
                              supporting_image_hard_prompt_file=file_root+"/supporting_image_hard_prompt_dict.json",
                              supporting_image_feature_file=file_root+"/supporting_image_global_feature_dict.pt",
                              synthetic_image_root=file_root+"/synthetic_images",
                              repharsing_image_root=file_root+"/supporting_images",
                              cap_rephrasing_cap_file=file_root+"/llm_rephrasing_cap.json",

                              tokenizer=captioner.tokenizer,
                              preprocess=image_preprocess)
    train_dataloader = DataLoader(train_data, 
                                  batch_size=args.batch_size, 
                                  num_workers=4, 
                                  shuffle=True, 
                                  drop_last=False,
                                  collate_fn=collate)
    
    out_dir=args.output_dir
    print("out_dir: ",out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_args_to_file(out_dir+"/train_args.json",args)

    

    optimizer = torch.optim.AdamW(captioner.parameters(),lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=args.epoch * len(train_dataloader)
    )
    

    for epoch in range(args.epoch):
        set_seed(args.seed)
        train_loss = []
        captioner.train()
        print(f">>> Training epoch {epoch}")
        progress = tqdm(total=len(train_dataloader))
        for image_global_feature, input_token, attention_mask, token_for_loss, image_instances in train_dataloader:
            image_global_feature = image_global_feature.to(device)
            image_instances = image_instances.to(device)
            input_token = input_token.to(device)
            attention_mask = attention_mask.to(device)
            token_for_loss = token_for_loss.to(device)
            image_grid_feature = clip_model.encode_image(image_instances).float()

            # construct hard prompt
            image_pj_feature=feature_project(image_global_feature,train_data.support_feature_norm.to(device)).float()
            soft_prompt = captioner.map_layer(image_pj_feature)

            output = captioner.language_decoder(input_ids=input_token, 
                                            encoder_hidden_states=image_grid_feature,
                                            attention_mask=attention_mask,
                                            labels=token_for_loss,
                                            prefix=soft_prompt)
            loss = output.loss
            train_loss.append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress.set_postfix({"loss =": np.mean(train_loss)})
            progress.update()
            
        progress.close()
        with open(out_dir+ "/train_log.txt", 'a+') as f:
            f.writelines('epoch ' + str(epoch) + ': ' + str(np.mean(train_loss)) + '\r\n')
        if args.save_per_model==True:
            torch.save(captioner.state_dict(),out_dir+ f"/captioner_{epoch}.pt")
            
    torch.save(captioner.state_dict(), out_dir+ f"/captioner_{epoch}.pt")

if __name__ == "__main__":
    args = get_args()
    main(args)