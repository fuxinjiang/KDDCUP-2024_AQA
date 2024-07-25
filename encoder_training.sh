cd sft

accelerate launch --main_process_port=25641 --config_file default_config.yaml encoder_training.py  \
--model_name_or_path "../model/Alibaba-NLP/gte-large-en-v1.5" \
--dataset "../data/process_data/pretrain_dataset.json" \
--output_dir "../output" \
--batch_size 24 \
--lr 1e-5 \
--epochs 5 \
--save_on_epoch_end 1 \
--gradient_accumulation_steps 24  \
--log_with 'wandb' \
--warmup_proportion 0.1 \
--neg_nums 5 \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \