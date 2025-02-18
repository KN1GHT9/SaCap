import sys
sys.path.insert(0,"../../SaCap")
import torch
import json
from clip import clip
from PIL import Image
import argparse
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mscoco")
    parser.add_argument("--test_dataset", type=str, default="mscoco")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args


def main(args):
    device = args.device
    data_root = f"../data/{args.dataset}/train"
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    with open(f"../data/{args.dataset}/train_captions.json") as f:
         captions = json.load(f)

    synthetic_image_pk_dict={i:cap for i,cap in enumerate(captions)}
    with open(f"../data/{args.dataset}/train/synthetic_image_pk_dict.json","w") as f:
         json.dump(synthetic_image_pk_dict,f)
    # Extract train caption feature
    caption_feature = dict()
    batch_size = args.batch_size
    if len(captions) % batch_size == 0:
        loop_num = len(captions)//batch_size
    else:
        loop_num = (len(captions)//batch_size) + 1
    with torch.no_grad():
        for i in tqdm(range(loop_num)):
            batch_caption = captions[i*batch_size:(i+1)*batch_size]
            clip_captions = clip.tokenize(
                batch_caption, truncate=True).to(device)
            clip_features = clip_model.encode_text(clip_captions)
            clip_features_norm = clip_features/clip_features.norm(dim=-1,keepdim=True)
            for caption, feature in zip(batch_caption, clip_features_norm):
                caption_feature[caption] = feature.cpu().float()
    torch.save(caption_feature, data_root+"/text_feature_dict.pt")


    # Extract object_level phrase text feature
    phrase_feature = dict()
    batch_size = args.batch_size
    with open(f"../data/object_phrases.json") as f:
        phrases = json.load(f)
    if len(phrases) % batch_size == 0:
        loop_num = len(phrases)//batch_size
    else:
        loop_num = (len(phrases)//batch_size) + 1
    with torch.no_grad():
        for i in tqdm(range(loop_num)):
            batch_phrases = phrases[i*batch_size:(i+1)*batch_size]
            clip_phrasess = clip.tokenize(
                batch_phrases, truncate=True).to(device)
            clip_features = clip_model.encode_text(clip_phrasess)
            clip_features_norm = clip_features/clip_features.norm(dim=-1,keepdim=True)
            for phrases, feature in zip(batch_phrases, clip_features_norm):
                phrase_feature[phrases] = feature.cpu().float()
    torch.save(phrase_feature, "../data/object_phrases_feature_dict.pt")


    # Extract test image feature
    refs_path = f"../data/{args.test_dataset}/refs.json"
    test_data_root=f"../data/{args.test_dataset}/test"
    test_img_root = f"../data/{args.test_dataset}/test/test_images/"
    with open(refs_path) as f:
        refs = json.load(f)
    test_img_names = list(refs.keys())
    test_img_path = [test_img_root+img_name +
                     '.jpg' for img_name in test_img_names]
    test_img_feature = dict()

    if len(test_img_names) % batch_size == 0:
        loop_num = len(test_img_names)//batch_size
    else:
        loop_num = (len(test_img_names)//batch_size) + 1
    with torch.no_grad():
        for i in tqdm(range(loop_num)):
            batch_img_names = test_img_names[i*batch_size:(i+1)*batch_size]
            batch_img_path = test_img_path[i*batch_size:(i+1)*batch_size]
            batch_img_instance = [preprocess(Image.open(img_path))
                                  for img_path in batch_img_path]
            batch_img_instance=torch.stack(batch_img_instance).to(device)


            img_features = clip_model.encode_image(batch_img_instance).float()
            img_features_norm=img_features/img_features.norm(dim=-1,keepdim=True)
            for img, feature in zip(batch_img_names, img_features_norm):
                test_img_feature[img] = feature.cpu().float()
    torch.save(test_img_feature, test_data_root+"/test_image_global_feature_dict.pt")


if __name__ == "__main__":
    args = get_args()
    main(args)
