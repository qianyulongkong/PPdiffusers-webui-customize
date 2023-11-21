import utils
from PIL import Image
import random
import os

# 文生图
def txt2img(model_name, lora_name, lora_style, prompt, negative_prompt, sampler, Image_size, guidance_scale, num_inference_steps, num_images, seed):

    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    txt2img = utils.txt2img(
        model_name=model_name, 
        lora_name=lora_name,
        lora_style=lora_style,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images, 
        scheduler_name=sampler,
        width=width,
        height=height,
        guidance_scale=float(guidance_scale),
        num_inference_steps=min(int(num_inference_steps), 100),
        max_embeddings_multiples=3,
        enable_parsing=True,
        seed=seed)

    tmp_path = []
    for idx, img in enumerate(txt2img):
        if not os.path.exists("./PPdiffusers-webui/output/output"):
            os.makedirs("./PPdiffusers-webui/output/output")
        save_path = os.path.join("./PPdiffusers-webui/output/output", lora_name + "_" + lora_style + "_" + f"{utils.out_put_num}_" + str(idx) + ".jpg")
        img.save(save_path)
        tmp_path.append(save_path)
    utils.out_put_num += 1
    return [Image.open(path) for path in tmp_path]

    
