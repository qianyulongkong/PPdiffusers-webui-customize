from contextlib import nullcontext
from pipeline import StableDiffusionPipelineAllinOne
from ppdiffusers import DPMSolverMultistepScheduler
from PIL import Image
import cv2
import numpy as np
import utils
from modules.helper import lora_helper
from modules.base_refactor import refactor_exchange
import paddle
import os
import json
import shutil
import zipfile

# # 基础模型，需要是paddle版本的权重，未来会加更多的权重
# pretrained_model_name_or_path = "/home/aistudio/PPdiffusers-webui/models/NovelAI_latest_ab21ba3c_paddle"
# # 我们加载safetensor版本的权重
# lora_outputs_path = "text_encoder_unet_lora.safetensors"
# # 加载之前的模型
# pipe = StableDiffusionPipelineAllinOne.from_pretrained(pretrained_model_name_or_path, safety_checker=None, feature_extractor=None,requires_safety_checker=False)
# # 设置采样器，采样器移到这里实现
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# # 加载lora权重, 可以选择加载和不加载lora, 没有lora时注释下行
# pipe.apply_lora(lora_outputs_path)
# pipe.apply_lora()

# 调度器
support_scheduler = [
    "EulerAncestralDiscrete",
    "PNDM",
    "DDIM",
    "LMSDiscrete",
    "HeunDiscrete",
    "KDPM2AncestralDiscrete",
    "KDPM2Discrete"
]

# 基底模型
model_name_list = [
    "Anything-v5.0", 
]

# 临时数据（待保存的角色名、角色示例图、角色Lora）
tmp_role_lora_state = "none"
tmp_role_image = "none"
tmp_role_name = "none"

# 角色名
role_name_list = [
    "none",
]

# 已保存数据（角色名、角色示例图、角色Lora）
role_name_img_list = [
    {"none":["./PPdiffusers-webui/models/Lora/Role_Lora/none_role.png"]},
]

# 临时数据（待保存的style名、style示例图、style Lora）
tmp_style_lora_state = "none"
tmp_style_image = "none"
tmp_style_name = "none"

# style名
style_name_list = [
    "none",
]

# 已保存数据（style名、style示例图、style Lora）
style_name_img_list = [
    {"none":["./PPdiffusers-webui/models/Lora/Style_Lora/none_style.jpg"]},
]


# 图片生成计数(哨兵)
out_put_num = 0

# 获取生成图片的尺寸
def get_size(standard_size):
    if standard_size == '512x768':
        width, height = 512, 768
    elif standard_size == '768x512':
        width, height = 768, 512
    elif standard_size == '512x512':
        width, height = 512, 512
    elif standard_size == '640x640':
        width, height = 640, 640
    elif standard_size == '自动判断':
        width, height = -1, -1
    else:
        width, height = 512, 512
    return width, height

context_null = nullcontext()

def ReadImage(image, height=None, width=None):
    """
    Read an image and resize it to (height,width) if given.
    If (height,width) = (-1,-1), resize it so that
    it has w,h being multiples of 64 and in medium size.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # clever auto inference of image size
    w, h = image.size
    if height == -1 or width == -1:
        if w > h:
            width = 768
            height = max(64, round(width / w * h / 64) * 64)
        else:  # w < h
            height = 768
            width = max(64, round(height / h * w / 64) * 64)
        if width > 576 and height > 576:
            width = 576
            height = 576
    if (height is not None) and (width is not None) and (w != width or h != height):
        image = image.resize((width, height), Image.ANTIALIAS)
    return image

# 训练角色特征提取Lora
def train_role_lora(model_name, train_dir, role_name, role_prompt):
    os.system(f'python -u ./PPdiffusers-webui/modules/base_train_lora.py \
    --pretrained_model_name_or_path="/home/aistudio/PPdiffusers-webui/models/{model_name}"  \
    --output_dir="./PPdiffusers-webui/models/Lora/Role_Lora/{role_name}"  \
    --train_data_dir={train_dir}  \
    --image_format="png" \
    --resolution=512  \
    --train_batch_size=1  \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=500 \
    --learning_rate=6e-5  \
    --report_to="visualdl" \
    --lr_scheduler="cosine_with_restarts"  \
    --lr_warmup_steps=0  \
    --max_train_steps=1000  \
    --lora_rank=128 \
    --validation_prompt={role_prompt} \
    --validation_epochs=1 \
    --validation_guidance_scale=5.0 \
    --use_lion False  \
    --seed=0')
    return "训练完成！"

def txt2img(model_name, lora_name, lora_style, prompt, scheduler_name, width, height, guidance_scale, num_inference_steps, negative_prompt, num_images_per_prompt, 
            max_embeddings_multiples, enable_parsing, fp16=False, seed=None):
    # scheduler = pipe.create_scheduler(scheduler_name)

    # 基础模型，需要是paddle版本的权重，未来会加更多的权重
    pretrained_model_name_or_path = "/home/aistudio/PPdiffusers-webui/models/" + model_name
    # 我们加载safetensor版本的权重
    if lora_name != "none":
        lora_name_path = "./PPdiffusers-webui/models/Lora/Role_Lora/" + lora_name + "text_encoder_unet_lora.safetensors"
    if lora_style != "none":
        lora_style_path = "./PPdiffusers-webui/models/Lora/Style_Lora/" + lora_style + "text_encoder_unet_lora.safetensors"
    # 加载之前的模型
    pipe = StableDiffusionPipelineAllinOne.from_pretrained(pretrained_model_name_or_path, safety_checker=None, feature_extractor=None,requires_safety_checker=False)
    # 设置采样器，采样器移到这里实现
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # 加载lora权重, 可以选择加载和不加载lora, 没有lora时选择none或注释下行
    if lora_name != "none":
        pipe.apply_lora(lora_name_path)
    if lora_style != "none":
        pipe.apply_lora(lora_style_path)
    # pipe.apply_lora()

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total)
        # print(i, total, tqdm_progess.format_dict)

    if fp16 and scheduler_name != "LMSDiscrete":
        context = paddle.amp.auto_cast(True, level='O2')
    else:
        context = context_null

    with context:
        return pipe.text2image(
            prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            num_images_per_prompt = num_images_per_prompt, 
            max_embeddings_multiples=int(max_embeddings_multiples),
            skip_parsing=(not enable_parsing),
            # scheduler=scheduler,
            callback=callback_fn,
        ).images


def img2img(model_name, lora_name, lora_style, image_path, prompt, scheduler_name, height, width, strength, num_inference_steps, guidance_scale,
            negative_prompt, num_images_per_prompt, max_embeddings_multiples, enable_parsing, fp16=True, seed=None):
    # scheduler = pipe.create_scheduler(scheduler_name)
    init_image = ReadImage(image_path, height=height, width=width)

    # 基础模型，需要是paddle版本的权重，未来会加更多的权重
    pretrained_model_name_or_path = "/home/aistudio/PPdiffusers-webui/models/" + model_name
    # 我们加载safetensor版本的权重
    if lora_name != "none":
        lora_name_path = "./PPdiffusers-webui/models/Lora/Role_Lora/" + lora_name + "text_encoder_unet_lora.safetensors"
    if lora_style != "none":
        lora_style_path = "./PPdiffusers-webui/models/Lora/Style_Lora/" + lora_style + "text_encoder_unet_lora.safetensors"
    # 加载之前的模型
    pipe = StableDiffusionPipelineAllinOne.from_pretrained(pretrained_model_name_or_path, safety_checker=None, feature_extractor=None,requires_safety_checker=False)
    # 设置采样器，采样器移到这里实现
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # 加载lora权重, 可以选择加载和不加载lora, 没有lora时选择none或注释下行
    if lora_name != "none":
        pipe.apply_lora(lora_name_path)
    if lora_style != "none":
        pipe.apply_lora(lora_style_path)
    # pipe.apply_lora()

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total)
        # print(i, total, tqdm_progess.format_dict)

    if fp16 and scheduler_name != "LMSDiscrete":
        context = paddle.amp.auto_cast(True, level='O2')
    else:
        context = context_null
    with context:
        return pipe.img2img(prompt,
                            seed=seed,
                            image=init_image,
                            num_inference_steps=num_inference_steps,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt,
                            num_images_per_prompt = num_images_per_prompt, 
                            max_embeddings_multiples=int(max_embeddings_multiples),
                            skip_parsing=(not enable_parsing),
                            # scheduler=scheduler,
                            callback=callback_fn
                            ).images


def inpaint(model_name, lora_name, lora_style, image_path, mask_path, prompt, scheduler_name, height, width, num_inference_steps, strength,
            guidance_scale, negative_prompt, num_images_per_prompt, max_embeddings_multiples, enable_parsing, fp16=True, seed=None):
    # scheduler = pipe.create_scheduler(scheduler_name)
    init_image = ReadImage(image_path, height=height, width=width)
    mask_image = ReadImage(mask_path, height=height, width=width)

    # 基础模型，需要是paddle版本的权重，未来会加更多的权重
    pretrained_model_name_or_path = "/home/aistudio/PPdiffusers-webui/models/" + model_name
    # 我们加载safetensor版本的权重
    if lora_name != "none":
        lora_name_path = "./PPdiffusers-webui/models/Lora/Role_Lora/" + lora_name + "text_encoder_unet_lora.safetensors"
    if lora_style != "none":
        lora_style_path = "./PPdiffusers-webui/models/Lora/Style_Lora/" + lora_style + "text_encoder_unet_lora.safetensors"
    # 加载之前的模型
    pipe = StableDiffusionPipelineAllinOne.from_pretrained(pretrained_model_name_or_path, safety_checker=None, feature_extractor=None,requires_safety_checker=False)
    # 设置采样器，采样器移到这里实现
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # 加载lora权重, 可以选择加载和不加载lora, 没有lora时选择none或注释下行
    if lora_name != "none":
        pipe.apply_lora(lora_name_path)
    if lora_style != "none":
        pipe.apply_lora(lora_style_path)
    # pipe.apply_lora()

    # 边运行的时候会边传递值到这里！
    def callback_fn(i, total, tqdm_progess):
        print(i, total)
        # print(i, total, tqdm_progess.format_dict)

    if fp16 and scheduler_name != "LMSDiscrete":
        context = paddle.amp.auto_cast(True, level='O2')
    else:
        context = context_null
    with context:
        return pipe.inpaint(
            prompt,
            seed=seed,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt, 
            max_embeddings_multiples=int(max_embeddings_multiples),
            skip_parsing=(not enable_parsing),
            # scheduler=scheduler,
            callback=callback_fn
        ).images

def refactor(init_image):
    out_put_path = f"./PPdiffusers-webui/output/input_refactor/refactor_{utils.out_put_num}.jpg"
    Image.fromarray(init_image).save(out_put_path)
    image = cv2.imread(out_put_path)
    save_path = f"./PPdiffusers-webui/output/output_refactor/refactor_{utils.out_put_num}.jpg"
    # 重构图片
    refactor_exchange(image, save_path)
    utils.out_put_num = utils.out_put_num + 1
    return save_path

# train dreambooth lora, aistudio project: https://aistudio.baidu.com/aistudio/projectdetail/5481677
# TODO 在训练中可视化输出的图到UI界面，拖动滑块查看不同时间步产生的结果
def train_dreambooth_lora(zip_file, pretrained_model_name, instance_prompt, validation_prompt, seed, instance_data_dir="./PPdiffusers-webui/models/Lora/Dream_booth_lora/train_dreambooth_lora", output_dir="./PPdiffusers-webui/models/Lora/Dream_booth_lora", resolution=512, train_batch_size=1,
                          gradient_accumulation_steps=1, checkpointing_steps=50, learning_rate=1e-4, report_to="visualdl", lr_scheduler="constant", lr_warmup_steps=0,
                          max_train_steps=100, lora_rank=128, validation_epochs=25, validation_guidance_scale=5.0, use_lion=False):
    # 基底模型存放文件夹路径（仅导入文件夹路径不导入model_name时，默认加载Anything-v5基底模型）
    pretrained_model_path = f"/home/aistudio/PPdiffusers-webui/models/{pretrained_model_name}"

    def unzip_file(zip_file):
        os.makedirs("./PPdiffusers-webui/models/Lora/Dream_booth_lora/train_dreambooth_lora", exist_ok=True)
        with zipfile.ZipFile(zip_file) as zip_ref:
            zip_ref.extractall('./PPdiffusers-webui/models/Lora/Dream_booth_lora/train_dreambooth_lora')

    def process_zip_file(zip_file):
        unzip_file(zip_file.name)
        print("文件已解压并处理完成。")
        # 在这里添加您的处理代码
        os.system(f'python ./PPdiffusers-webui/modules/train_dreambooth_lora.py \
          --pretrained_model_name_or_path={pretrained_model_path}  \
          --instance_data_dir="{instance_data_dir}" \
          --output_dir={output_dir} \
          --instance_prompt={instance_prompt} \
          --resolution={resolution} \
          --train_batch_size={train_batch_size} \
          --gradient_accumulation_steps={gradient_accumulation_steps} \
          --checkpointing_steps={checkpointing_steps} \
          --learning_rate={learning_rate} \
          --report_to={report_to} \
          --lr_scheduler={lr_scheduler} \
          --lr_warmup_steps={lr_warmup_steps} \
          --max_train_steps={max_train_steps} \
          --lora_rank={lora_rank} \
          --validation_prompt={validation_prompt} \
          --validation_epochs={validation_epochs} \
          --validation_guidance_scale={validation_guidance_scale} \
          --use_lion {use_lion} \
          --seed={seed}')
        return "训练完成！"

    return process_zip_file(zip_file)

# 训练style Lora
def train_style_lora(model_name, train_dir, style_name, style_prompt, style_seed):
    # 判断是否已有正要训练的角色对应的文件夹，如果有，则删除
    if os.path.exists("./PPdiffusers-webui/models/Lora/Style_Lora/" + style_name):  
        shutil.rmtree("./PPdiffusers-webui/models/Lora/Style_Lora/" + style_name)
    os.system(f'python -u ./PPdiffusers-webui/modules/base_train_lora.py \
    --pretrained_model_name_or_path="/home/aistudio/PPdiffusers-webui/models/{model_name}"  \
    --output_dir="./PPdiffusers-webui/models/Lora/Style_Lora/{style_name}"  \
    --train_data_dir={train_dir}  \
    --image_format="png" \
    --resolution=512  \
    --train_batch_size=1  \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=500 \
    --learning_rate=6e-5  \
    --report_to="visualdl" \
    --lr_scheduler="cosine_with_restarts"  \
    --lr_warmup_steps=0  \
    --max_train_steps=1000  \
    --lora_rank=128 \
    --validation_prompt={style_prompt} \
    --validation_epochs=1 \
    --validation_guidance_scale=5.0 \
    --use_lion False  \
    --seed={style_seed}')
    return "训练完成！"
