ngpus=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

nvidia-smi --query-gpu=name --format=csv,noheader

WORLD_SIZE=1
NPROC_PER_NODE=$ngpus
MASTER_PORT=`shuf -i 2000-65000 -n 1`

RANK=0

LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$((256 / (LOCAL_BATCH_SIZE * ngpus)))
echo "Number of GPUs: $ngpus"
echo "GPU Batch Size: $LOCAL_BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"

OUTP_DIR=models/hackathon_models/
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'

RUN_NAME=olmo2_vqa_1b
export WANDB_NAME=$RUN_NAME


torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    src/lmms/finetune.py \
    --deepspeed src/lmms/deepspeed/stage2.json \
    --config olmo2_vqa_1b_shadow \
    --method fft \
    --visual_proj_ckpt_dir models/vp/olmo_1b_vp_tsvqa/non_lora_trainables.bin \
    --pretrained_ckpt_path None \
    --bf16 True \
    --tasks "p4ms_vqa_shadow,llava_vqa,synthdog_en,ocrvqa,text_ocr,text_caps" \
    --output_dir $OUTP_DIR/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters False \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 1 \
    --dataloader_persistent_workers True \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --report_to none \
    --seed 1234

