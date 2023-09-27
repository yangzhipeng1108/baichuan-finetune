import torch
import os

model_static_dict = {}
for path in os.path.isfile(pipeline_model_dir).iterdir():
    print("已经处理文件：{}".format(path))
    if not path.name.startswith('layer'):
        continue
    small_static_dict = torch.load(path, map_location="cpu")
    layer_i = int(path.name.split('-')[0].replace('layer_', ''))
    if layer_i == 0:
        model_static_dict["transformer.word_embeddings.weight"] = small_static_dict["word_embeddings.weight"]
    elif layer_i == 30:
        model_static_dict["lm_head.weight"] = small_static_dict["word_embeddings.weight"]
    elif layer_i == 29:
        for k, v in small_static_dict.items():
            model_static_dict["transformer." + k] = v
    else:
        for k, v in small_static_dict.items():
            model_static_dict["transformer." + k.replace("layer.", "layers.{}.".format(layer_i - 1))] = v

torch.save(model_static_dict, os.path.join(save_model_dir, "pytorch_model.bin"))


# model = ChatGLMForConditionalGeneration.from_pretrained(args.model_name_or_path)
# model_pipe = PipelineModule(layers=get_model(model), num_stages=args.num_stages)
# engine, _, _, _ = deepspeed.initialize(model=model_pipe, config=ds_config, model_parameters=model_pipe.parameters())
