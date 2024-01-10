
export PATH=/root/miniconda3/envs/NegInstruct/bin:$PATH
threshold=1.1
weight_unlike=1
name=cut-1plus-13b
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1233 finetune_unlikelihood.py \
    --base_model saved_models/llama2-13b-chat-hf \
    --data-path data/iter/train-alpaca-sample-iter1.json \
    --output_dir ./saved_models/lora/${name} \
    --batch_size 8 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 0.0004 \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100\
    --weight_unlike ${weight_unlike}\
    --threshold ${threshold}\
    --downsample 0.25\

CUDA_VISIBLE_DEVICES=0 python merge.py \
    --base_model_name_or_path saved_models/llama2-13b-chat-hf \
    --peft_model_path ./saved_models/lora/${name} \
    --output_dir ./saved_models/${name}
