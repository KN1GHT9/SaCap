from diffusers import StableDiffusionPipeline
import torch
import json
from tqdm import tqdm
import os
import argparse
from PIL import Image,ImageFile
from clip import clip
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="mscoco")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--diffusion_step", type=int, default=20)
    parser.add_argument("--split_part", type=int, default=0)
    args = parser.parse_args()
    return args


def generate_syn_img(args):
    device = args.device
    image_root = f"../data/{args.dataset}/train/supporting_images/"

    if not os.path.exists(image_root):
        os.makedirs(image_root)
    exist_data=list(map(lambda x:x.replace(".jpg",""),os.listdir(image_root)))
    
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    with open(f"../data/{args.dataset}/train/llm_rephrasing_cap.json") as f:
        cap_recap = json.load(f)
        
    supporting_img_pk_map={}
    for cap in cap_recap:
        for recap in cap_recap[cap]:
            supporting_img_pk_map[len(supporting_img_pk_map)]=recap
    with open(f"../data/{args.dataset}/train/supporting_image_pk_dict.json","w") as f:
        json.dump(supporting_img_pk_map,f)

    for key in exist_data:
        if key in supporting_img_pk_map:
            supporting_img_pk_map.pop(key)
    data = list(supporting_img_pk_map.items())
    batch = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(int(len(data)/batch)+1)):

            batch_data = data[i*batch:(i+1)*batch]
            index_list = [item[0] for item in batch_data]
            conditional_texts = [item[1] for item in batch_data]

            images = pipe(conditional_texts, generator=generator,
                          num_inference_steps=args.diffusion_step).images
            for index, img in zip(index_list, images):
                img.save(image_root+f"/{index}.jpg")


def extract_syn_img_feature(args):
    device=args.device
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    clip_model=clip_model.to(device).eval()
    image_root = f"../data/{args.dataset}/train/supporting_images/"
    with open(f"../data/{args.dataset}/train/supporting_image_pk_dict.json") as f:
        sdf_num_caption_map=json.load(f)
    res_dict={}
    img_captions=list(sdf_num_caption_map.values())
    imgs_path=[image_root+str(i)+".jpg" for i in sdf_num_caption_map.keys()]

    batch_size=256
    if len(imgs_path)%batch_size==0:
        iter_num=len(imgs_path)//batch_size
    else:
        iter_num=(len(imgs_path)//batch_size)+1

    for i in tqdm(range(iter_num)):
        with torch.no_grad():
            batch_img_path=imgs_path[i*batch_size:(i+1)*batch_size]
            batch_img_instance = [preprocess(Image.open(img_path))
                                  for img_path in batch_img_path]
            batch_img_instance=torch.stack(batch_img_instance).to(device)
            batch_caption=img_captions[i*batch_size:(i+1)*batch_size]

            img_features = clip_model.encode_image(batch_img_instance).float()
            img_features_norm=img_features/img_features.norm(dim=-1,keepdim=True)
            for caption, feature in zip(batch_caption, img_features_norm):
                res_dict[caption] = feature.cpu().float()
    torch.save(res_dict,f"../data/{args.dataset}/train/supporting_image_global_feature_dict.pt")

if __name__ == "__main__":
    args = get_args()
    generate_syn_img(args)
    extract_syn_img_feature(args)
