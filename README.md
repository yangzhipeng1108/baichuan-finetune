
## baichuan微调
本项目主要针对baichuan和baichuan2模型进行不同方式的微调（Freeze方法、Lora方法、P-Tuning方法、全量参数等），并对比大模型在不同微调方法上的效果，主要针对信息抽取任务、生成任务、分类任务等。

本项目支持单卡训练&多卡训练，由于采用单指令集方式微调，模型微调之后**并没有出现严重的灾难性遗忘**。

由于官方代码和模型一直在更新，目前代码和模型使用的是最新版本（20230806）。

PS：没有用Trainer（虽然Trainer代码简单，但不易修改，大模型时代算法工程师本就成为了数据工程师，因此更需了解训练流程）

## 微调方法
模型微调时，如果遇到显存不够的情况，可以开启gradient_checkpointing、zero3、offload等参数来节省显存。

下面model_name_or_path参数为模型路径，请根据可根据自己实际模型保存地址进行修改。
### Freeze方法
Freeze方法，即参数冻结，对原始模型部分参数进行冻结操作，仅训练部分参数，以达到在单卡或多卡，不进行TP或PP操作就可以对大模型进行训练。

微调代码，见train.py，核心部分如下：
```python3
freeze_module_name = args.freeze_module_name.split(",")
for name, param in model.named_parameters():
	if not any(nd in name for nd in freeze_module_name):
		param.requires_grad = False
```
针对模型不同层进行修改，可以自行修改freeze_module_name参数配置，例如"layers.27.,layers.26.,layers.25.,layers.24."。
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、freeze_module_name、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

baichuan单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path baichuan-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode baichuan \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-baichuan
```
baichuan四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path baichuan-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode baichuan \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-baichuan
```
baichuan2单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path baichuan2-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode baichuan \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-baichuan
```
baichuan2四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path baichuan2-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode baichuan \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-baichuan
```
