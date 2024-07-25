## Data Downlod  
mkdir data  
cd data  
将AQA数据集放置在data/AQA下  
## Environment
python3.10  
pip install -r requirements.txt  

## Data preprocessing  
cd src  
python3 data_preprocessing.py  

## Training Data & Negative Samples
cd src 
python3 negative_samples.py  

## Fine-tuning
bash encoder_training.sh

## Build Retriever
cd src  
python3 build_retriever.py  

## get related_doc
cd src  
python3 get_related_doc.py  

## rrf
cd src  
python3 rrf.py    