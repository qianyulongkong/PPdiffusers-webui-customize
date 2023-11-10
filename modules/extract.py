import utils
from PIL import Image
import random
import os
import numpy as np
import cv2
import shutil

import paddle
from ppdiffusers import StableDiffusionPipeline


# 提取特征(训练角色Lora)
def extract_img(model_name, extract_image_init, role_name, role_prompt):
    out_dir = "./PPdiffusers-webui/models/Lora/Role_Lora/" + role_name
    if os.path.exists(out_dir):
        # 清空inputs文件夹 
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir = out_dir + "/pictures"
    os.makedirs(out_dir, exist_ok=True)

    file_paths = [file_d['name'] for file_d in extract_image_init]
    input_dir = "./PPdiffusers-webui/models/Lora/Role_Lora/tmp_picture"

    #自动生成所使用的tag
    initializer_tag = role_prompt
        
    # 遍历新上传的文件
    for file_path in file_paths:
        # 获取文件名
        filename = os.path.basename(file_path)
        # 从tmp_picture构建输出文件的完整路径
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(out_dir, filename)
        # 读取图片
        img = cv2.imread(file_path)
        # 保存图片
        if img is not None:
            cv2.imwrite(output_path, img)
        with open(output_path.split(".")[0]+".txt", "w", encoding="utf-8") as f:
            f.write(initializer_tag)
            f.close()
        # print(output_path)

    # 训练代码,得到角色Lora模型,返回训练状态
    tmp_role_lora_state = utils.train_role_lora(model_name, out_dir, role_name, role_prompt)
    utils.tmp_role_lora_state = tmp_role_lora_state
    # 将第一张图片用作展示
    # 读取第一张图片，在右下角标记角色名，并另存为label_image.jpg
    out_label_image = "./PPdiffusers-webui/models/Lora/Role_Lora/" + role_name + "/label_image.jpg"
    img = cv2.imread(os.path.join(input_dir, os.path.basename(file_paths[0])))
    # # 为了更好地展示，将示例图片重设大小为(294,294),因为web-ui上展示框是294x294
    # img = cv2.resize(img, (294, 294))

    # 获取图片的高度和宽度
    height, width = img.shape[:2]

    # 设置文字的属性
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = role_name
    fontScale = 1  # 字体大小
    color = (107, 114, 128)  # 文字颜色，这里是白色
    thickness = 1  # 文字粗细

    # 获取字体尺寸
    font_shape = cv2.getTextSize(text, font, fontScale, thickness)
    font_width = font_shape[0][0]     # 文本基准点与右边界之间的距离
    # font_height = font_shape[0][1]     # 文本基准点与上边界之间的距离
    # font_bottom = font_shape[1]     # 文本基准点与下边界之间的距离
    # 字体白色背景块
    white_ground = np.zeros_like(img[height - 33:, width - font_width - 20:, :]) + 255
    img[height - 33:, width - font_width - 20:, :] = white_ground

    # 这是文字的位置，可以根据需要调整
    position = (width - font_width - 10, height - 10)
    # 在图片右下角添加文字(暂不支持中文)
    image_with_text = cv2.putText(img, text, position, font, fontScale, color, thickness, cv2.LINE_AA)
  
    if img is not None:
        cv2.imwrite(out_label_image, image_with_text)
    # 得到label_image.jpg路径
    utils.tmp_role_image = out_label_image
    # 得到角色名
    utils.tmp_role_name = role_name

# 导入/更新角色特征
def induct_update():
    if utils.tmp_role_lora_state == "none":
        # print("导入失败！请先提取角色特征！！")
        pass

    # 逻辑上该判断条件多余，但未删除
    elif utils.tmp_role_name in utils.role_name_list:
        role_idx = utils.role_name_list.index(utils.tmp_role_name)

        tmp_role_dict = {utils.tmp_role_name:[utils.tmp_role_image]}
        utils.role_name_img_list[role_idx] = tmp_role_dict
        
        # 避免多次加入相同的角色Lora
        utils.tmp_role_lora_state = "none"
        utils.tmp_role_image = "none"
        utils.tmp_role_name = "none"

    else:
        # 以角色名命名保存对应Lora模型，并将Lora名加入utils.role_name_img_list对应角色字典中

        tmp_role_dict = {utils.tmp_role_name:[utils.tmp_role_image]}
        utils.role_name_list.append(utils.tmp_role_name)
        utils.role_name_img_list.append(tmp_role_dict)

        # 避免多次加入相同的角色Lora
        utils.tmp_role_lora_state = "none"
        utils.tmp_role_image = "none"
        utils.tmp_role_name = "none"
    
    # # 角色名列表
    # display_roles_name = [name for role in utils.role_name_img_list for name in (list(role.keys()))]
    # 图片路径列表
    display_roles_path = [path for role in utils.role_name_img_list for path in (list(role.values())[0])]
    return display_roles_path


# 清空角色列表
def clear():
    utils.role_name_img_list = [
                {"none":["./PPdiffusers-webui/models/Lora/Role_Lora/none_role.png"]}
            ]
    # # 角色名列表
    # display_roles_name = [name for role in utils.role_name_img_list for name in (list(role.keys()))]
    # 图片路径列表
    display_roles_path = [path for role in utils.role_name_img_list for path in (list(role.values())[0])]
    return display_roles_path


