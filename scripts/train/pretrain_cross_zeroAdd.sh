#!/bin/bash

# # for vicuna
# model_name_or_path="/workspace/vicuna-13b-v1.5"
# version="plain"

# for qwen2
model_name_or_path="/workspace/Qwen2-7B-Instruct"
version="qwen_2"

# # for llama3
# model_name_or_path="/workspace/Meta-Llama-3-8B-Instruct"
# version="llama_3"


vision_tower="/workspace/clip-vit-large-patch14-336"
text_tower="/workspace/clip-vit-large-patch14-336"

output_dir="/workspace/checkpoints/tg_llava_pretrain"

CMDARG="--deepspeed scripts/zero2.json \
        --model_name_or_path $model_name_or_path \
        --version $version \
        --data_path /workspace/blip_laion_cc_sbu_558k.json \
        --image_folder /workspace/images/pretrain \
        --vision_tower $vision_tower \
        --text_tower $text_tower \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 True \
        --output_dir $output_dir \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 0.5 \
        --save_total_limit 2 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none
"

echo "$CMDARG"

deepspeed --num_gpus=4 tg_llava/train/train_mem_cross_zeroAdd.py $CMDARG