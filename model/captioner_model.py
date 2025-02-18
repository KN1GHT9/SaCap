import sys
sys.path.insert(0,"../../SaCap")
import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import pickle


class Mlp(nn.Module):
    def __init__(self,input_dim=512,hidden_dim=2048,output_dim=768,prefix_len=1):
        super().__init__()
        self.prefix_len=prefix_len
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim*self.prefix_len)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if len(x.shape)==3:
            x=x.reshape(x.shape[0],x.shape[1],self.output_dim)
        else:
            x=x.reshape(x.shape[0],self.prefix_len,self.output_dim)
        return x

class Captioner(nn.Module):
    def __init__(self,prefix_len=1):
        super(Captioner,self).__init__()
        self.language_decoder,self.tokenizer = self.load_weight_from_gpt('./model/decoder_config.pkl')
        self.map_layer=Mlp(prefix_len=prefix_len)
    

    def load_weight_from_gpt(self,config_path):
        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./model/gpt2")
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

        # add cross-attention adapter layer via configure file
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        gpt2_model_added_adapter = GPT2LMHeadModel(config)

        cross_atten_param_name=gpt2_model_added_adapter.transformer.load_state_dict(torch.load("./model/gpt2/pytorch_model.bin"),strict=False)[0]
        return gpt2_model_added_adapter, gpt2_tokenizer
    
    
    
    def batch_caption_generation(self,image_pk,input_token, attention_mask, image_grid_feature, max_generate_length, soft_prompt):
        generate_res={}
        output = self.language_decoder.generate(input_ids=input_token, 
                                    attention_mask=attention_mask, 
                                    encoder_hidden_states=image_grid_feature,
                                    eos_token_id=self.tokenizer.encode(".")[0],
                                    max_new_tokens=max_generate_length,
                                    prefix=soft_prompt,
                                    do_sample=False,
                                    num_beams=5)
        generated_texts = self.tokenizer.batch_decode(output)
        res_texts = [output.split(":")[-1].split(".")[0].lower()+"." for output in generated_texts]
        for img, cap in zip(image_pk, res_texts):
            generate_res[img] = cap
        return generate_res