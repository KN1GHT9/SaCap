python data_prepare.py \
    --dataset "mscoco" \
    --test_dataset "mscoco" \
    --device "cuda:0"

python image_synthesis.py \
    --dataset "mscoco" \
    --device "cuda:0"

python caption_rephrasing.py \
    --dataset "mscoco" \
    --device "cuda:0"

python supporting_image_synthesis.py \
    --dataset "mscoco" \
    --device "cuda:0"

python hard_prompt_retrieval.py \
    --dataset "mscoco" \
    --device "cuda:0"
 