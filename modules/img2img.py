import utils
from PIL import Image
import random
import os

# 图生图
def img2img(model_name, lora_name, lora_style, init_image, prompt, negative_prompt, num_images, sampler, Image_size, strength, guidance_scale, num_inference_steps, seed):

    width, height = utils.get_size(Image_size)
    seed = random.randint(0, 2 ** 32) if seed == '-1' else int(seed)

    Image.fromarray(init_image).save(f"./PPdiffusers-webui/output/input_img2img/tem_{lora_name}_{utils.out_put_num}.jpg")
    img2img = utils.img2img(
        model_name=model_name, 
        lora_name=lora_name, 
        lora_style=lora_style, 
        image_path="./PPdiffusers-webui/output/input_img2img/tem_{lora_name}_{utils.out_put_num}.jpg",
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images, 
        scheduler_name=sampler,
        width=width,
        height=height,
        strength=float(strength),
        num_inference_steps=min(int(num_inference_steps), 100),
        guidance_scale=float(guidance_scale),
        max_embeddings_multiples=3,
        enable_parsing=True,
        seed=seed,
        fp16=False)

    tmp_path = []
    for idx, img in enumerate(img2img):
        save_path = os.path.join("./PPdiffusers-webui/output/output", lora_name + "_" + lora_style + "_" + f"{utils.out_put_num}_" + str(idx) + ".jpg")
        img.save(save_path)
        tmp_path.append(save_path)
    utils.out_put_num += 1
    return tmp_path
