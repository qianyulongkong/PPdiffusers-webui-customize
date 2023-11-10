# 👾二次元角色风格定制✨👾

二次元角色风格定制是一款基于Paddle框架实现的深度学习模型工具。用户仅需要输入提示词或额外提供一张照片即可获得某一角色对应风格的图片。

如果不满于现有内置的角色和风格，也可以输入10-30张图片训练自己的角色、风格Lora，进行组合使用。

如果仅仅想体验一下该项目的效果，只需要看完应用篇就可以轻松使用。想要完整学习本项目，请阅览进阶篇。在进阶篇，你将熟悉PPdiffusers的使用、文生图、图生图的项目结构以及Lora的训练与应用等等。

有什么问题可以在评论区提出，我会尽力回答。期待你有所收获。


# 一、应用篇

## 1.拉取代码仓库

拉取github上可能因为网络原因导致失败，我们可以拉取gitee上的仓库。gitee与github仓库保持同步更新。


```python
# 克隆gitee上代码，-b表示特定分支，我们选择ppdiffusers分支
!git clone -b ppdiffusers 

"""# 克隆github上代码，-b表示特定分支，我们选择ppdiffusers分支
!git clone -b ppdiffusers 
```

    正克隆到 'PPdiffusers-webui'...
    remote: Enumerating objects: 2249, done.[K
    remote: Counting objects: 100% (2249/2249), done.[K
    remote: Compressing objects: 100% (1407/1407), done.[K
    remote: Total 2249 (delta 1322), reused 1732 (delta 808), pack-reused 0[K
    接收对象中: 100% (2249/2249), 10.03 MiB | 2.32 MiB/s, 完成.
    处理 delta 中: 100% (1322/1322), 完成.
    检查连接... 完成。





    '# 克隆github上代码，-b表示特定分支，我们选择ppdiffusers分支\n!git clone -b ppdiffusers https://github.com/tensorfly-gpu/PPdiffusers-webui'



## 2.解压模型文件

解压Anything-v5.0模型。也可选择下载模型文件，将新模型放在models文件夹下，并将名称加入utils.py中的model_name_list即可。

model_name_list = ["Anything-v5.0", 
            "新模型文件夹名称"
            ]


```python
!unzip data/data*/Anything-v5.0.zip -d PPdiffusers-webui/models
```

## 3.运行launch.py配置环境

运行launch.py配置环境，配置完之后，重启内核。

**运行以下代码，然后重启内核即可**


```python
%cd /home/aistudio/PPdiffusers-webui
!python launch.py
```

## 4.进入UI界面

进入UI界面。双击 PPdiffusers-webui/launch.gradio.py 文件。

在右边出现UI界面后，点击下图中红框位置，即可使用网页进行打开。然后就可以进行AI绘画了！

![](https://ai-studio-static-online.cdn.bcebos.com/c00d41e0ab7d4ababacdd8728f56271f33c2058d71514927b00c2fccc7024d3a)


# 二、进阶篇

## 1、项目结构

二次元角色风格定制项目主要分为前端和后端两个结构：

**前端部分：**使用第三方库Gradio构建前端界面，并主要使用按钮（Gradio.button）的click函数与后端进行连接。

**后端部分：**本项目中后端共有约六个模块：角色特征提取、文生图、图生图、局部重绘、超分重构、训练。其中，角色特征提取本质上就是训练每一个角色的Lora模型。文生图、图生图、局部重绘这三部分是应用了ppdiffusers内置的接口实现。超分重构暂未实现（预计将realcu GAN迁移到paadle使用）。训练部分包含dreambooth lora、style lora两部分。

## 2、目录结构

项目目录文件结构如下：

![](https://ai-studio-static-online.cdn.bcebos.com/a683b75c29d24c13965f3b580b0a74e4fee541f36e2746718c5a5e4de34dbb63)


## 3、数据传输，在每个文件、模块间的流通

项目文件、模块间的数据流如下：

![](https://ai-studio-static-online.cdn.bcebos.com/9d87c6b7f781488ea3a3f2f2c028f356b8af4b98192b45e18846d734691c7663)


## 4、Gradio界面UI

本项目的界面代码均在webui.py中，经由launch.gradio.py启动。

**界面UI如下：**

界面展示视频链接：【二次元角色风格定制项目界面UI展示】 https://www.bilibili.com/video/BV1fu4y187CJ

**webui代码介绍如下：**

![](https://ai-studio-static-online.cdn.bcebos.com/5dd3702c368345a1aef88d40faeba2bfb4379817fd984c1d8c9b9d0a15e93b0c)


![](https://ai-studio-static-online.cdn.bcebos.com/1fabf55bc8a2425f809959a821fce97734e0129cd4bb48a59e6b24cb2354b650)



## 5、核心模块utils

utils.py文件是本项目中的一大核心模块，它就像整个项目的数据中转站。

![](https://ai-studio-static-online.cdn.bcebos.com/38717706d837480bb48a96a59c15ff4c54b0515691734873ac58012d3a8e5a3e)


## 6、模块详解
### 6.1、角色特征提取

![](https://ai-studio-static-online.cdn.bcebos.com/b17ab3bad27f405298d6840457b8b89e6c507e39028b469e89854f8f6a0b3273)


### 6.2、文生图

![](https://ai-studio-static-online.cdn.bcebos.com/018f4a5e0be64f44801a1ace8320488fbd1ebb82d6784d628c375bf5e0c4e839)


### 6.3、图生图

![](https://ai-studio-static-online.cdn.bcebos.com/b9dce240baf64cb086de95ea546726c50f59f7b534ef47f4a0f2aff34e51344b)


### 6.4、局部重绘

![](https://ai-studio-static-online.cdn.bcebos.com/c57781f79a3c4170acf55b17fa14bffb9c389ad37a704e6fafcd2a746baea955)


### 6.5、超分重构

**详见utils.py介绍**

### 6.6、训练

#### 6.6.1、Dreambooth Lora训练

**详见utils.py介绍**

#### 6.6.2、Lora训练

![](https://ai-studio-static-online.cdn.bcebos.com/cde349c7a6904c5d89e475732192e8387d846a101138404cba32d9ed43f1744c)



# 三、关于本项目

本项目使用ppdiffusers作为后端推理，gradio作为前端页面，深度学习框架是paddle。

## 1、目前待修复的内容：
1. 未加入ControlNet
1. 角色特征提取方案未确定，有两个方案，1为每个角色训练一个lora，2训练一个通用的Lora提取角色特征（已选择第一种方案）
1. 选择角色、风格时是否展示对应图片
1. 文生图、图生图、局部重绘刷新按钮未定义选项刷新函数（刷新按钮暂不能用）
1. 上传图片名不能有中文（能保存到项目文件夹但无法在界面预览）
1. 角色名部分暂时不支持中文（角色展框会显示？？？）
1. extract部分中tmp_role_lora_state设置Lora训练完成进度条提示（tmp_role_lora_state与训练脚本异步运行，暂时不能代表训练状态）
1. 训练过程中未加入训练动态进度条
1. tmp_style_lora部分中tmp_style_lora_state的问题同extract部分
1. 使用多个Lora模型（非模型融合）方案：去ppiffusers中抄一份自定义的Lora层，然后搞个多Lora层
1. 将utils存储各选项的变量中的常量转为从文件读取。（为每个选项创建一个存储值的txt文件，避免下次运行环境时各选项重置又无法导入已有的值）,别忘了图片生成计数(哨兵)
1. ppdiffusers/stable diffusion基底模型微调过程
## 2、参考项目：
1. https://aistudio.baidu.com/projectdetail/5766153
1. https://aistudio.baidu.com/projectdetail/5964987
1. https://aistudio.baidu.com/projectdetail/5540885
1. https://aistudio.baidu.com/projectdetail/6790638
1. https://aistudio.baidu.com/projectdetail/6129764
1. https://aistudio.baidu.com/projectdetail/6926368
## 3、其他项目：暂无
