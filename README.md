# AI Writer Web


使用 AI 实现文章续写。项目基于魔改的 GPT 模型，在不牺牲效果的前提下，将传统 GPT 模型的硬件门槛降低了1000倍，实现写作机器人的定制。本项目为 AI Writer 的 Web 端，通过调用已训练好的模型快速体验 AI 续写效果。

>- AI Writer 的 [Model 项目](https://www.oneflow.cloud/drill/#/project/public/code?id=95d6894ae425d4f8ecd8c3e998df2344)
>- 参考项目 [BlinkDL/AI-Writer](https://github.com/BlinkDL/AI-Writer)

## 1. 效果示意
![web](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/web.gif)



## 2. 目录结构

```
.
|-- README.md                      # 项目说明文件
|-- model                          # 模型文件夹
    |-- wangwen                    # 基于网文训练好的模型
    `-- wangwen.json               # 网文词典
|-- src                            # 实现 Web 端推理的核心功能文件
|-- static                         # Web 端所需静态文件
|-- templates                      # html 文件夹
    |-- error.html                 # 500 时的错误提示页面
    |-- index.html                 # 主页面
    `-- notFound.html              # 404 时的错误提示页面
|-- pictures                       # 相关图片
|-- requirements.txt               # 项目相关依赖
|-- infer.py                       # 加载训练好的模型并进行推断的工具类
|-- config.py                      # 配置文件
|-- server.py                      # Web 服务端
`-- init.sh                        # 部署 Web 端环境的初始化文件
```



## 3. 使用说明

项目支持快速推理。直接 `Fork` 项目，在 **在线推理** 模块进行部署具体步骤如下。

在项目详情页，点击【Fork】。

<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/1-Fork.png" alt="Fork" style="zoom:50%;" />

弹出 `Fork` 项目基本信息，用户可编辑，一般默认即可，点击【提交】即成功 `Fork` 项目。此时项目成为 `我的项目`，用户可以修改或运行调试算法。
- `运行` 是指启动支持项目运行的服务器，以便用户进行调试。调试方式详见 [帮助文档](https://www.oneflow.cloud/drill/#/help)。需要注意的是运行服务器后，附件中的文件会自动复制到的 `我的项目` 的 `/workspace` 目录下。AI 实训平台为每个用户云端持久化存储空间，访问路径为项目下 `/workspace`，**只有** 写到该目录下的文件或安装程序包将被持久化存储，用户在重启任务时文件不会初始化重置丢失。
- `部署` 是指启动以网页形式提供当前项目服务的服务器，此服务器不可以访问，但是可以通过 `在线推理` 页面 `日志` 查看服务启动信息。

<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/2-my_project.png" alt="my_project" style="zoom:50%;" />

对于当前项目而言，我们只需要进行部署即可。点击 `部署` 按钮，在弹窗中选择项目文件夹，点击 `下一步` 按钮。

<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/3-deploy.png" alt="deplay" style="zoom:50%;" />

随后进入 `基本信息` 填写页，这一页填写的内容自定义即可。

<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/4-base_info.png" alt="base_info" style="zoom:50%;" />

最后进入 `配置信息` 填写页，这一页内容比较重要，按如下方式填写。

```
启动命令行：sh init.sh && python server.py
端口：5000
其他：随意
```
<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/deployment.png" alt="deployment" style="zoom:50%;" />

此时查看日志文件，可以看到模型运行产生的日志。

<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/5-log.png" alt="deployment" style="zoom:50%;" />

点击 `测试` 或者直接复制 API，即可以访问到 AI Writer 的网页界面。该页面支持功能：

- 自定义续写长度 n （默认为10）
- 续写：在用户输入的文本基础上进行长度为 n 的文本续写
- 重写：对已完成的续写进行重写
- 启动日志和报错信息提醒（页面右下方）
