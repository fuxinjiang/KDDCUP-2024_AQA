# 环境搭建
python3.7  
pip install -r requirements.txt  

# 模型下载
1. 首先在[huggingface]{https://huggingface.co/} 下载nvidia/NV-Embed-v1存到model文件夹  

# 数据下载
1. 将下载的数据放到src文件夹  

# 构造retriever
1. python3 build_retriever.py  

# 生成embedding
1. python3 get_emb.py  

# 生成最后结果
1. python3 get_result.py  111
