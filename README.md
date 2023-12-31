## baichuan 针对 Belle数据Lora微调
### Baichuan2-13B-Chat微调lora 模型数据
https://huggingface.co/yangzp1108/baichuan-finetune/tree/main

## baichuan微调
本项目主要针对baichuan和baichuan2模型进行不同方式的微调（Freeze方法、Lora方法、全量参数等），并对比大模型在不同微调方法上的效果，主要针对信息抽取任务、生成任务、分类任务等。

本项目支持单卡训练&多卡训练，由于采用单指令集方式微调，模型微调之后**并没有出现严重的灾难性遗忘**。

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
                --mode baichuan2 \
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
                --mode baichuan2 \
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
                --mode baichuan2 \
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
                --mode baichuan2 \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-baichuan
```


### Lora方法
Lora方法，即在大型语言模型上对指定参数（权重矩阵）并行增加额外的低秩矩阵，并在模型训练过程中，仅训练额外增加的并行低秩矩阵的参数。
当“秩值”远小于原始参数维度时，新增的低秩矩阵参数量也就很小。在下游任务tuning时，仅须训练很小的参数，但能获取较好的表现结果。

![](images/Lora.png)
- 论文：[paper](https://arxiv.org/abs/2106.09685)
- 官方代码：[Github](https://github.com/microsoft/LoRA)
- HuggingFace封装的peft库：[Github](https://github.com/huggingface/peft)

微调代码，见train.py，核心部分如下：
```python3
model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
lora_module_name = args.lora_module_name.split(",")
config = LoraConfig(r=args.lora_dim,
					lora_alpha=args.lora_alpha,
					target_modules=lora_module_name,
					lora_dropout=args.lora_dropout,
					bias="none",
					task_type="CAUSAL_LM",
					inference_mode=False,
					)
model = get_peft_model(model, config)
model.config.torch_dtype = torch.float32
```
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、lora_dim、lora_alpha、lora_dropout、lora_module_name、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

baichuan单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path baichuan-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode baichuan2 \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj" \
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
              --model_name_or_path baichuan-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode baichuan2 \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj" \
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
              --model_name_or_path baichuan2-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode baichuan2 \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj" \
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
              --model_name_or_path baichuan2-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode baichuan2 \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj" \
              --seed 1234 \
              --ds_file ds_zero2_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-baichuan
```


注意：Lora方法在模型保存时仅保存了Lora训练参数，因此在模型预测时需要将模型参数进行合并，具体参考merge_lora.py。

### 全参方法
全参方法，对大模型进行全量参数训练，主要借助DeepSpeed-Zero3方法，对模型参数进行多卡分割，并借助Offload方法，将优化器参数卸载到CPU上以解决显卡不足问题。

微调代码，见train.py，核心部分如下：
```python3
model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
```
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

baichuan四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path baichuan-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode baichuan2 \
              --train_type all \
              --seed 1234 \
              --ds_file ds_zero3_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-baichuan
```
baichuan2四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path baichuan2-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode baichuan2 \
              --train_type all \
              --seed 1234 \
              --ds_file ds_zero3_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-baichuan
```


### 运行环境
查看requirements.txt文件
