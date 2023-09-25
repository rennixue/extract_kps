# extract_kps
提取关键词


### 1、安装依赖包
* pip install -r requirements.txt

* en_core_web_sm 需要单独安装
* 进入package目录使用pip install en_core_web_sm-3.6.0-py3-none-any.whl 命令进行安装

### 2、代码介绍
* extract_origin.py 为wws所写代码，代码已跑通

* utils.py为更改后的代码

### 3、使用说明
* 调用utils.py 里的main函数传入pdf文件所在路径，会提取出关键词，并在同级目录下生成关键词对应的csv文件
### 4、优化方向
* 调整关键词输出数量
* 优化textrank处理文档的长度
* 数据处理优化
