#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import random
import shutil
import gradio as gr
import os
import cv2
import utils
from PIL import Image
from modules import txt2img, img2img, inpaint
# from modules.extract import extract_img, induct_update, clear
from modules.extract import extract_img, induct_update, clear
from modules.train_style_lora import train_style_lora, induct_style_update
import zipfile

# controlnet #
# from examples.controlnet.gradio_pose2image import process as pose2image


def upload_role_file(files, current_files):
    out_dir = './PPdiffusers-webui/models/Lora/Role_Lora/tmp_picture'
    if os.path.exists(out_dir):
        # 清空inputs文件夹 
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    file_paths_Deduplication = []
    # file_paths去重
    for file_path in file_paths:
        if file_path not in file_paths_Deduplication:
            file_paths_Deduplication.append(file_path)
    # 遍历新上传的文件
    for file_path in file_paths_Deduplication:
        # 获取文件名
        filename = os.path.basename(file_path)
        # 构建输出文件的完整路径
        output_path = os.path.join(out_dir, filename)
        # 读取图片
        img = cv2.imread(file_path)
        # 保存图片
        if img is not None:
            cv2.imwrite(output_path, img)
        print(output_path)

    return file_paths_Deduplication

def upload_style_file(files, current_files):
    out_dir = './PPdiffusers-webui/models/Lora/Style_Lora/tmp_picture'
    if os.path.exists(out_dir):
        # 清空inputs文件夹 
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    file_paths_Deduplication = []
    # file_paths去重
    for file_path in file_paths:
        if file_path not in file_paths_Deduplication:
            file_paths_Deduplication.append(file_path)
    # 遍历新上传的文件
    for file_path in file_paths_Deduplication:
        # 获取文件名
        filename = os.path.basename(file_path)
        # 构建输出文件的完整路径
        output_path = os.path.join(out_dir, filename)
        # 读取图片
        img = cv2.imread(file_path)
        # 保存图片
        if img is not None:
            cv2.imwrite(output_path, img)
        print(output_path)

    return file_paths_Deduplication

## UI设计 ##
with gr.Blocks() as demo:
    # 顶部文字
    gr.Markdown("""
    ## 【AI绘画】 👾二次元角色风格定制✨👾
    ##### 介绍：基于PPdiffusers二次元角色风格定制教程  二次元角色名+ 描述文本+ 1张照片即可获得对应二次元角色的各种风格图片
    ##### 开源项目地址：
    """)

    # 多个tab
    with gr.Tabs():
        with gr.TabItem("角色特征提取"):
            with gr.Column():
                model_name = gr.Dropdown(utils.model_name_list, label="请选择一个基底模型", default="Anthing-v5.0", multiselect=False)
                with gr.Row():
                    extract_image_init = gr.Gallery(label="请上传图片", elem_id="gallery").style(columns=[5], object_fit="contain", height="auto")

                with gr.Row():
                    upload_button = gr.UploadButton("上传", file_types=["image"], file_count="multiple")
                    clear_button = gr.Button("清空")

                    clear_button.click(fn=lambda: [], inputs=None, outputs=extract_image_init)
                    upload_button.upload(fn=upload_role_file, inputs=[upload_button, extract_image_init], outputs=extract_image_init, queue=False)
                with gr.Row():
                    role_name = gr.Textbox(label="role_name", placeholder="请输入角色名", interactive=True, value=None)
                    role_prompt = gr.Textbox(label="prompt(instance and validation)", lines=1, placeholder="请输入提示词")
                # 生成、重置按钮（row：行）
                with gr.Row():
                    extract_img_button = gr.Button("提取")
                    induct_update_button = gr.Button("导入/更新")
                with gr.Row():
                    clear_button = gr.Button("重置角色列表")
                with gr.Row():
                    # # 角色名列表
                    # display_roles_name = [name for role in utils.role_name_img_list for name in (list(role.keys()))]
                    # 图片路径列表
                    display_roles_path = [path for role in utils.role_name_img_list for path in (list(role.values())[0])]
                    display_boxes = gr.Gallery(value=display_roles_path, label="角色列表", elem_id="gallery").style(columns=[5], object_fit="contain", height="auto")
                    # # 另一种尚未完善的展示方式（废弃）
                    # with gr.Column():
                        # # 角色展示框
                        # with gr.Row():
                        #     gr.Markdown("""
                        #     ###### 角色列表
                        #     """)
                        # display_boxes = utils.role_name_img_list
                        # roles = display_boxes.copy()
                        # boxes_num = len(display_boxes)
                        # # 每行展示5个角色
                        # # 计算一共多少行
                        # rows = math.ceil(boxes_num / 5)
                        # for row in range(rows):
                        #     if row == rows - 1:
                        #         fu_num = 5 - len(display_boxes[row * 5:])
                        #         with gr.Row():
                        #             for box in display_boxes[row * 5:]:
                        #                     roles[roles.index(box)] = gr.Image(value=list(box.values())[0][0] if list(box.values())[0][0] != "none" else None, height=64, width=64, label=list(box.keys())[0], type="pil")
                        #             for fu_box in range(fu_num):
                        #                 with gr.Row():
                        #                     pass
                        #     else:
                        #         with gr.Row():
                        #             for box in display_boxes[row * 5:row * 5 + 5]:
                        #                     roles[roles.index(box)] = gr.Image(value=list(box.values())[0][0] if list(box.values())[0][0] != "none" else None, height=64, width=64, label=list(box.keys())[0], type="pil")

        with gr.TabItem("文生图"):
            with gr.Column():
                with gr.Row():
                    model_name1 = gr.Dropdown(utils.model_name_list, label="请选择一个基底模型", default="Anthing-v5.0", multiselect=False)
                with gr.Row():
                    lora_name1 = gr.Dropdown(utils.role_name_list, label="请选择角色", default="none", multiselect=False)
                # 待修复，加入风格选项
                with gr.Row():
                    lora_style1 = gr.Dropdown(utils.style_name_list, label="请选择风格", default="none", multiselect=False)
                with gr.Row():

                    txt2img_prompt = gr.Textbox(label="prompt", lines=3, placeholder="请输入正面描述", interactive=True, value=None)
                    txt2img_negative_prompt = gr.Textbox(label="negative_prompt", lines=2, placeholder="请输入负面描述", interactive=True, value=None)

                    with gr.Row():
                        txt2img_sampler = gr.Dropdown(utils.support_scheduler, label="Sampling method", default="DDIM", multiselect=False)
                        txt2img_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Sampling steps", interactive=True)

                    with gr.Row():
                        txt2img_Image_size = gr.Dropdown(["512x512", "512x768", "768x512", "640x640"], default="512x768", label="Image size", multiselect=False)
                        txt2img_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                    txt2img_num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num images", interactive=True)
                    txt2img_cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.1, label="CFG Scale", interactive=True)

                    # 生成、刷新按钮（row：行）
                    with gr.Row():
                        txt2img_button = gr.Button("生成")
                        refrash_button = gr.Button("刷新")

                        # # 刷新按钮
                        # refrash_button.click()

                with gr.Row():
                    # 输出框
                    txt2img_output = gr.Gallery(label="Image").style(columns=3)

        with gr.TabItem("图生图"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        model_name2 = gr.Dropdown(utils.model_name_list, label="请选择一个基底模型", default="Anthing-v5.0", multiselect=False)
                        with gr.Row():
                            img2img_prompt = gr.Textbox(label="prompt", lines=2, placeholder="请输入正面描述", interactive=True,
                                            value=None)
                            img2img_negative_prompt = gr.Textbox(label="negative_prompt", lines=2, placeholder="请输入负面描述",
                                            interactive=True, value=None)
                        with gr.Row():
                            lora_name2 = gr.Dropdown(utils.role_name_list, label="请选择角色", default="none", multiselect=False)
                        # 待修复，加入风格选项
                        with gr.Row():
                            lora_style2 = gr.Dropdown(utils.style_name_list, label="请选择风格", default="none", multiselect=False)
                        with gr.Row():
                            with gr.Column():
                                img2img_sampler = gr.Dropdown(utils.support_scheduler, label="Sampling method", default="DDIM",
                                                    multiselect=False)
                                img2img_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Sampling steps",
                                                interactive=True)
                            with gr.Column():
                                img2img_Image_size = gr.Dropdown(["512x512", "512x768", "768x512", "640x640", "自动判断"], default="512x768",
                                                        label="Image size", multiselect=False)
                                img2img_num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num images", interactive=True)
                            with gr.Column():
                                img2img_cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.1, label="CFG Scale", interactive=True)
                                img2img_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="strength",
                                                interactive=True)
                                img2img_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")
                            img2img_image_init = gr.Image(label="请上传图片")

                        # 生成、刷新按钮（row：行）
                        with gr.Row():
                            img2img_button = gr.Button("生成")
                            refrash_button2 = gr.Button("刷新")

                            # # 刷新按钮
                            # refrash_button2.click()

                with gr.Row():
                    # 输出框
                    img2img_output = gr.Gallery(label="Image").style(columns=3)

        with gr.TabItem("局部重绘"):
            with gr.Row():
                with gr.Column(): 
                    model_name3 = gr.Dropdown(utils.model_name_list, label="请选择一个基底模型", default="Anthing-v5.0", multiselect=False)
                    with gr.Row():
                        lora_name3 = gr.Dropdown(utils.role_name_list, label="请选择角色", default="none", multiselect=False)
                    # 待修复，加入风格选项
                    with gr.Row():
                        lora_style3 = gr.Dropdown(utils.style_name_list, label="请选择风格", default="none", multiselect=False)

                    inpaint_prompt = gr.Textbox(label="prompt", lines=3, placeholder="请输入正面描述", interactive=True,
                                        value=None)
                    inpaint_negative_prompt = gr.Textbox(label="negative_prompt", lines=2, placeholder="请输入负面描述",
                                                 interactive=True, value=None)

                    with gr.Row():
                        inpaint_sampler = gr.Dropdown(utils.support_scheduler, label="Sampling method", default="DDIM",
                                              multiselect=False)
                        inpaint_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Sampling steps",
                                          interactive=True)

                    with gr.Row():
                        inpaint_Image_size = gr.Dropdown(["512x512", "512x768", "768x512", "640x640", "自动判断"], default="512x768",
                                                 label="Image size", multiselect=False)
                        # 每次允许重绘图片的数量为1
                        inpaint_num_images = gr.Slider(minimum=1, maximum=1, value=1, step=1, label="Num images", visible=False, interactive=True)
                        # num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Num images", interactive=True)

                        inpaint_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                    with gr.Row():
                        inpaint_cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, step=0.1, label="CFG Scale", interactive=True)
                        inpaint_strength = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="strength",
                                          interactive=True)
                    refrash_button3 = gr.Button("刷新")
                    # # 刷新按钮
                    # refrash_button3.click()

                with gr.Column():
                    # 上传图片，生成图片
                    with gr.Row():
                        inpaint_image_mask_init = gr.Image(tool="sketch", label="请上传图片，然后使用鼠标涂抹",
                                                           type="numpy")
                    # 输出框
                    with gr.Row():
                        inpaint_output = gr.Image(type="pil")
                    with gr.Row():
                        inpaint_button = gr.Button("生成")

        with gr.TabItem("超分重构"):
            with gr.Row():
                with gr.Column():
                    cf_input = gr.Image(label="原图")
                with gr.Column():
                    cf_output = gr.Image(label="超分重构结果")
            cf_button = gr.Button("超分辨率重构")
        with gr.TabItem("训练"):
            with gr.Row():
                model_name4 = gr.Dropdown(utils.model_name_list, label="请选择一个基底模型", default="Anthing-v5.0", multiselect=False)
            # 多个tab
            with gr.Tabs():
                # 训练第一种lora
                with gr.TabItem("train dreambooth lora"):
                    # 使用1行2列
                    with gr.Row():
                        with gr.Column():
                            # dataset通过解压上传的压缩包上传时，同时启动训练

                            # TODO: 其他参数设置

                            file_upload = gr.File()
                            # 输出框
                            output_text = gr.Textbox(label="训练状态")

                            train_dreambooth_lora_button = gr.Button("开始训练")

                        with gr.Column():
                            dreambooth_prompt = gr.Textbox(label="prompt(instance and validation)", lines=1, placeholder="请输入提示词")
                            dreambooth_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

                with gr.TabItem("train style lora"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                style_image_init = gr.Gallery(label="请上传图片", elem_id="gallery").style(columns=[5], object_fit="contain", height="auto")

                            with gr.Row():
                                upload_style_button = gr.UploadButton("上传", file_types=["image"], file_count="multiple")
                                clear_style_button = gr.Button("清空")

                                clear_style_button.click(fn=lambda: [], inputs=None, outputs=style_image_init)
                                upload_style_button.upload(fn=upload_style_file, inputs=[upload_style_button, style_image_init], outputs=style_image_init, queue=False)
                            train_style_lora_button = gr.Button("开始训练")
                            induct_style_lora_button = gr.Button("导入选项")
                            # 输出框
                            output_style_text = gr.Textbox(label="训练状态")

                        with gr.Column():
                            style_name = gr.Textbox(label="style", placeholder="请输入风格名称", interactive=True)
                            style_prompt = gr.Textbox(label="prompt(instance and validation)", lines=1, placeholder="请输入提示词")
                            style_seed = gr.Textbox(label="seed", default="-1", lines=1, placeholder="请输入种子，默认-1")

        # with gr.TabItem("ControlNet"):
        #     with gr.Row():
        #         with gr.Column():
        #             input_image = gr.Image(source="upload", type="numpy")
        #             hand = gr.Checkbox(label="detect hand", value=False)
        #             prompt = gr.Textbox(label="Prompt")
        #             pose2image_button = gr.Button(label="Run")
        #             with gr.Accordion("Advanced options", open=False):
        #                 num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
        #                 image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512,
        #                                              step=64)
        #                 strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0,
        #                                      step=0.01)
        #                 guess_mode = gr.Checkbox(label="Guess Mode", value=False)
        #                 detect_resolution = gr.Slider(
        #                     label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1
        #                 )
        #                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
        #                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
        #                 seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
        #                 eta = gr.Number(label="eta (DDIM)", value=0.0)
        #                 a_prompt = gr.Textbox(label="Added Prompt", value="best quality, extremely detailed")
        #                 n_prompt = gr.Textbox(
        #                     label="Negative Prompt",
        #                     value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        #                 )
        #         with gr.Column():
        #             result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(
        #                 grid=2, height="auto"
        #             )
        #     ips = [
        #         input_image,
        #         hand,
        #         prompt,
        #         a_prompt,
        #         n_prompt,
        #         num_samples,
        #         image_resolution,
        #         detect_resolution,
        #         ddim_steps,
        #         guess_mode,
        #         strength,
        #         scale,
        #         seed,
        #         eta,
        #     ]

    extract_img_button.click(fn=extract_img,
                      inputs=[model_name, extract_image_init, role_name, role_prompt])

    induct_update_button.click(fn=induct_update, inputs=None, outputs=display_boxes)

    clear_button.click(fn=clear, inputs=None, outputs=display_boxes)

    txt2img_button.click(fn=txt2img,
                      inputs=[model_name1, lora_name1, lora_style1, txt2img_prompt, txt2img_negative_prompt, txt2img_sampler, txt2img_Image_size, txt2img_cfg_scale, txt2img_steps, txt2img_num_images, txt2img_seed],
                      outputs=txt2img_output)

    img2img_button.click(fn=img2img,
                         inputs=[model_name2, lora_name2, lora_style2, img2img_image_init, img2img_prompt, img2img_negative_prompt, img2img_num_images, img2img_sampler, img2img_Image_size, img2img_strength, img2img_cfg_scale, img2img_steps, img2img_seed],
                         outputs=img2img_output)

    inpaint_button.click(fn=inpaint,
                         inputs=[model_name3, lora_name3, lora_style3, inpaint_image_mask_init, inpaint_prompt, inpaint_negative_prompt, inpaint_num_images, inpaint_sampler, inpaint_Image_size, inpaint_strength, inpaint_cfg_scale, inpaint_steps, inpaint_seed],
                         outputs=inpaint_output)

    cf_button.click(fn=utils.refactor, inputs=cf_input, outputs=cf_output)

    train_dreambooth_lora_button.click(
        fn=utils.train_dreambooth_lora,
        inputs=[file_upload, model_name4, dreambooth_prompt, dreambooth_prompt, dreambooth_seed],
        outputs=output_text)

    train_style_lora_button.click(
        fn=train_style_lora,
        inputs=[model_name4, style_image_init, style_name, style_prompt, style_seed],
        outputs=output_style_text)

    induct_style_lora_button.click(fn=induct_style_update, inputs=None, outputs=None)

    # pose2image_button.click(fn=pose2image, inputs=ips, outputs=[result_gallery])
