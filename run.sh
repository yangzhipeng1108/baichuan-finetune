deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29000 train.py \
                --train_path output.jsonl   \
                --model_name_or_path /root/llama1/rl_dpo/Baichuan2-7B-Chat \
                --per_device_train_batch_size 2 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode baichuan2 \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json  \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-glm2
