# SaCap

A PyTorch implementation of "Synthesize-then-Align: Modality Alignment Augmentation for Image Captioning with Synthetic Data".
## Requirements

Install the requirements with:
```bash
$ pip install -r requirements.txt
```
## Dataset
Download image caption datasets from the web. The data directory looks like:

~~~bash
data
├── mscoco
│   ├── train_captions.json		#captions of training split
│   ├── refs.json		        #reference of test split
│   └── test					
│       └── test_image			#images of test split
├── flickr
│   ├── train_captions.json		#captions of training split
│   ├── refs.json		        #reference of test split
│   └── test					
│       └── test_image			#images of test split
├── ss1m
│   └── train_captions.json		#captions of training split
├── cc3m
│   └── train_captions.json		#captions of training split
├── nocaps
│   ├── refs.json               #reference of test split
│   └── test                    #images of test split
│       └── test_image
└── object_phrases.json			#object phrases used to construct hard prompt
~~~

## Pretrained Model
Download pretrained [GPT2](https://huggingface.co/openai-community/gpt2/) from Huggingface into the `./model` directory. [Stable Diffusion v1.5]([(https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)]), [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and [CLIP VIT-B/32](https://github.com/openai/CLIP) can be downloaded automatically within code. The directory looks like:

  ```bash
  model
  ├── gpt2
  |   └── ...    # Pretrained GPT2
  ```


## Data Preparation
We perform data preparation before proceeding with training, and realize this process by running the code under the `./data_preparation` directory.
* **Extract CLIP features for training and test**

  ```bash
  python data_prepare.py --dataset {mscoco|flickr30k|ss1m|cc3m}
  ```
  The following file is obtained after executing the above command, using MSCOCO as an example.

  ~~~bash
  data
  ├── mscoco
  │   ├── train
  │   │   ├── text_feature_dict.pt		  		#clip text feature of captions
  │   │   ├── synthetic_image_pk_dict.json		#synthetic pseudo pair's map
  │   │   └── ...
  │   └── test
  │       ├── test_image_global_feature_dict.pt	#clip image feature of test images
  │       └── ...
  ├── object_phrases_feature_dict.pt		  		#clip text feature of object phrases
  ~~~

* **Image synthesis**
  This section generates synthetic images for all conditional texts from the given corpus.

  ```bash
  python image_synthesis.py --dataset {mscoco|flickr30k|ss1m|cc3m}
  ```
  The following directory or file is obtained after executing the above command, using MSCOCO as an example.

  ~~~bash
  data
  ├── mscoco
  │   ├── train
  │   │   ├── synthetic_images						#synthetic image of captions
  │   │   ├── synthetic_image_global_feature_dict.pt	#image global feature of synthetic images
  │   │   └── ...
  ~~~

* **Rephrasing caption**
  ```bash
  python caption_rephrasing.py --dataset {mscoco|flickr30k|ss1m|cc3m}
  ```
  The following file is obtained after executing the above command, using MSCOCO as an example.

  ~~~bash
  data
  ├── mscoco
  │   ├── train
  │   │   ├── llm_rephrasing_cap.json       #texts from rephrasing
  │   │   └── ...
  ~~~

* **supporting image synthesis**
  This section generates supporting images for texts from rephrasing.

  ```bash
  python supporting_image_synthesis.py --dataset {mscoco|flickr30k|ss1m|cc3m}
  ```
  The following directory or file is obtained after executing the above command, using MSCOCO as an example.

  ~~~bash
  data
  ├── mscoco
  │   ├── train
  │   │   ├── supporting_images			        	#supporting image of text from rephrasing
  │   │   ├── supporting_image_global_feature_dict.pt	#image global feature of supporting images
  │   │   ├── supporting_image_pk_dict.json          	#supporting image name and conditional text map
  │   │   └── ...
  ~~~

* **Constructing hard prompts**
  This section retrieves the **Top-N** support features with the highest similarity to the target feature based on the cosine similarity of the CLIP features. It is used to retrieve relevant object phrases.

  ```bash
  python hard_prompt_retrieval.py --dataset {mscoco|flickr30k|ss1m|cc3m}
  ```
  The following directory or file is obtained after executing the above command, using MSCOCO as an example.
  
  ~~~bash
  data
  ├── mscoco
  │   ├── train
  │   │   ├── synthetic_image_hard_prompt_dict.json    #shard Prompt of supporting images
  │   │   ├── supporting_image_hard_prompt_dict.json	 #hard Prompt of synthetic images
  │   │   └── ...
  │   ├── test
  │   │   ├── test_image_hard_prompt_dict.json	     #hard Prompt of test images
  │   │   └── ...
  ~~~


## Training
We use a re-pairing mechanism to construct training pairs for each iteration and augment the cross-modal alignment modeling with the soft prompt and hard prompt. 
Running the following code will create a folder `./trained_model/{dataset}` in the root directory, and save the training log, argument, and model weights.

  ```bash
  python training.py --dataset {mscoco|flickr30k|ss1m|cc3m}
  ```
  The following folder or file is obtained after executing the above command, using MSCOCO as an example.

~~~bash
trained_model
├── mscoco
|   ├──captioner_{epoch}.pt        #trained model 
|   ├──train_log.txt               #training log 
|   └──train_args.json             #training args
~~~

## Inference
Executing the following command performs `in-domain`, `cross-domain` or `zero-shot` experiments by setting `--test_dataset`.
  ```bash
  python inference.py 
  --model_path {trained model}               #trained model 
  --test_dataset {mscoco|flickr30k|nocaps}   #target domain 
  ```
  The inference results file will be obtained in the same directory as the model weights file, using MSCOCO as an example.
  ```bash
trained_model
├── mscoco
|   ├──captioner_{epoch}.pt							#trained model 
|   ├──captioner_{epoch}_{test_dataset}_res.json    #inference result 
|   └──...
  ```

## Evaluation
Input the inference result file and the corresponding `refs.json` to compute the validation metrics.
```bash
python eval_metrics.py
--candidates_json {inference result}        #inference result
--references_json {reference file}          #reference file
```

