mkdir alpaca
cd alpaca

# get convert_llama_weights_to_hf.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py -O convert_llama_weights_to_hf.py

# download llama
# covert llamma weight to hf version
git lfs install
git clone https://huggingface.co/nyanko7/LLaMA-7B llama_ckpt/7B
cd llama_ckpt
ln -s 7B/tokenizer.model .
cd ..
python convert_llama_weights_to_hf.py --input_dir llama_ckpt --model_size 7B --output_dir llama_hf_ckpt

# get alpaca-7b-wdiff
git lfs install
git clone https://huggingface.co/tatsu-lab/alpaca-7b-wdiff

# generate alpaca ckpt
git clone git@github.com:tatsu-lab/stanford_alpaca.git
python stanford_alpaca/weight_diff.py recover --path_raw llama_hf_ckpt --path_diff alpaca-7b-wdiff --path_tuned alpaca_ckpt
