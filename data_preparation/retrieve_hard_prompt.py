import sys
sys.path.insert(0,"../../SaCap")
import torch
import json
from tqdm import tqdm
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mscoco")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--topk", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args


def retrieve_phrase(source_feature_dict,support_data_dict,topq):
    res_list=[]
    device=args.device
    source_data_pk=list(source_feature_dict.keys())
    source_features=torch.vstack(list(source_feature_dict.values())).to(device)
    support_data_pk=list(support_data_dict.keys())
    support_features=torch.vstack(list(support_data_dict.values())).to(device)

    with torch.no_grad():
        batch_size=500
        if len(source_features)%batch_size==0:
            iter_num=len(source_features)//batch_size
        else:
            iter_num=len(source_features)//batch_size+1
        for i in tqdm(range(iter_num)):        
            batch_source_features = source_features[i*batch_size:(i+1)*batch_size].to(device)
            batch_source_data_pk=source_data_pk[i*batch_size:(i+1)*batch_size]
            
            sim = batch_source_features @ support_features.T.float()
            values,indices=sim.topk(topq)

            for source_pk,sim_nouns,sim_value in zip(batch_source_data_pk,indices,values):
                temp_dict={}
                temp_dict["pk"]=source_pk
                temp_dict["hard_prompt"]=[support_data_pk[index] for index in sim_nouns]
                temp_dict["sim_value"]=sim_value.detach().cpu().tolist()
                res_list.append(temp_dict)
    return res_list




if __name__ == "__main__":
    args = get_args()
    object_phrase_feature_dict=torch.load("../data/object_phrases_feature_dict.pt")

    synthetic_image_global_feature_dict=torch.load(f"../data/{args.dataset}/train/synthetic_image_global_feature_dict.pt")
    sythetic_image_hard_prompt_dict=retrieve_phrase(synthetic_image_global_feature_dict,object_phrase_feature_dict,args.topk)
    with open(f"../data/{args.dataset}/train/synthetic_image_hard_prompt_dict.json","w") as f:
        json.dump(sythetic_image_hard_prompt_dict,f)
    
    supporting_image_global_feature_dict=torch.load(f"../data/{args.dataset}/train/supporting_image_global_feature_dict.pt")
    supporting_image_hard_prompt_dict=retrieve_phrase(supporting_image_global_feature_dict,object_phrase_feature_dict,args.topk)
    with open(f"../data/{args.dataset}/train/supporting_image_hard_prompt_dict.json","w") as f:
        json.dump(supporting_image_hard_prompt_dict,f)

    test_image_global_feature_dict=torch.load(f"../data/{args.dataset}/test/test_image_global_feature_dict.pt")
    supporting_image_hard_prompt_dict=retrieve_phrase(test_image_global_feature_dict,object_phrase_feature_dict,args.topk)
    with open(f"../data/{args.dataset}/test/test_image_hard_prompt_dict.json","w") as f:
        json.dump(supporting_image_hard_prompt_dict,f)
