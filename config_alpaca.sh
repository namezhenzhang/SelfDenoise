#!/bin/bash

mkdir alpaca
cd alpaca
# get convert_llama_weights_to_hf.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py -O convert_llama_weights_to_hf.py

#download llama
mkdir llama_ckpt
mkdir llama_ckpt/7B
wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model -O ./llama_ckpt/tokenizer.model
wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk -O ./llama_ckpt/tokenizer_checklist.chk
wget https://agi.gpt4.org/llama/LLaMA/7B/consolidated.00.pth -O ./llama_ckpt/7B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/7B/params.json -O ./llama_ckpt/7B/params.json
wget https://agi.gpt4.org/llama/LLaMA/7B/checklist.chk -O ./llama_ckpt/7B/checklist.chk

# covert llamma weight to hf version
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 convert_llama_weights_to_hf.py \
    --input_dir llama_ckpt --model_size 7B --output_dir llama_hf_ckpt

# get alpaca-7b-wdiff
git clone https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
cd alpaca-7b-wdiff
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/95005cba09e22aa4cb30927810a673d7deffed7b7fa14067217626990fa1a114?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model-00001-of-00003.bin%3B+filename%3D%22pytorch_model-00001-of-00003.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1685201476&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzY4LzA5LzY4MDk2MmM1ZWVkMTczYjFlMjQyYjE2OWRlMGI1NTQ1NGI3Y2Q3YWQwNGU1YzkzYjVkYjhkYjhkYWM5ZWM2OGMvOTUwMDVjYmEwOWUyMmFhNGNiMzA5Mjc4MTBhNjczZDdkZWZmZWQ3YjdmYTE0MDY3MjE3NjI2OTkwZmExYTExND9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODUyMDE0NzZ9fX1dfQ__&Signature=Y7ry5qLQAZe8wqTmccQbcnRLGVeUEADobIjBNTeWi10B42sNeqXjkcoOd7HwQmyOBDQhDiYUdaylTNRkGzyu3WsxA6pCytYI%7EYhGr-eYjtkfI4MvuyVhV%7EYKQtl6ReuxWi1ll1lvQmISLLM6hflxLpneSDAXH6E34kp%7E-Z0OWzphqC4IqzkIBjtEqGgf-H8ah92VbOT4aFvRA5mS9R-yZTU-qJyLIw62nF4pI983f%7EZcyYrAiUaK7XnaCTnfVaGa6gOWqKE-%7E8IBf7MYuVifKY3CvJP0WrZrkBYZHb-HM%7EjkZ1NUGJaknDsLa0fPED4Hk1H8dnJwrIwxsPb3ztbH%7EQ__&Key-Pair-Id=KVTP0A1DKRTAX" -O pytorch_model-00001-of-00003.bin
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/cf76cc78bec93ccbbc18ecbb686d08d4916aa00781bf68457805c9ba0761974b?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model-00002-of-00003.bin%3B+filename%3D%22pytorch_model-00002-of-00003.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1685201561&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzY4LzA5LzY4MDk2MmM1ZWVkMTczYjFlMjQyYjE2OWRlMGI1NTQ1NGI3Y2Q3YWQwNGU1YzkzYjVkYjhkYjhkYWM5ZWM2OGMvY2Y3NmNjNzhiZWM5M2NjYmJjMThlY2JiNjg2ZDA4ZDQ5MTZhYTAwNzgxYmY2ODQ1NzgwNWM5YmEwNzYxOTc0Yj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODUyMDE1NjF9fX1dfQ__&Signature=RX9cp0xtrBdWQB1Iqt4%7EhMfNGUDsW6r8uC8iYAxqGIE%7EKcAYJGliu1HJaXsD4gWRWohK7RFJxYuipHc7GHMPowCQ8eiBwc-cB9IvpWGSE-YWojWOkMNvFEjJra1ahnFcXAplYlZ8jyFs9xNFzCRy-cA7h1e4pflmb1SplazwOT%7E75q5m1wkPViPU2qf0fgRS9p4QZuSKThmWPpU6momYBlOltqAwpYrV-mb5-Dj-CzYni-5gIZkU1dV8UjTkUvwsNBei%7EEkiQz1fqwkHcpL2jjuAM%7ElZFCZfG620hMGsGbMom82xEsJKRl4HaIEI0Vp-0jl-Fm9Owxamv9zIy-fdUg__&Key-Pair-Id=KVTP0A1DKRTAX" -O pytorch_model-00002-of-00003.bin
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/6f04361aac986bd6031b5f6ae8db2af30cb2e7edd792d9c1ee4074e47c9608eb?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model-00003-of-00003.bin%3B+filename%3D%22pytorch_model-00003-of-00003.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1685201607&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzY4LzA5LzY4MDk2MmM1ZWVkMTczYjFlMjQyYjE2OWRlMGI1NTQ1NGI3Y2Q3YWQwNGU1YzkzYjVkYjhkYjhkYWM5ZWM2OGMvNmYwNDM2MWFhYzk4NmJkNjAzMWI1ZjZhZThkYjJhZjMwY2IyZTdlZGQ3OTJkOWMxZWU0MDc0ZTQ3Yzk2MDhlYj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODUyMDE2MDd9fX1dfQ__&Signature=A%7EyuErMV8bsDMzT2kCOxyEfucEJJCNb6fhd7qmNqDGggEV6eVIzmxXpiSbXGVRFiNt7HvnDkeInCfU0Mma1pErHw%7EJEsRg7PaTjgAiR8vritNVxn-ODrFWiEBJhWg7D-Ogd6kxtKQtGd6kAxHkeCV11LFYjMVgpAxKP-MOWQKDW6ZVZtImOcwMdPViH32IuiKish634K0uUlWFCuQTnqGCWa8Z1-eBKkve5CTDct5%7EVXieAdGVUe0JWm%7EEeWNK5MWJP2hFe3%7EB-5ErPIu9dDc%7E7X21XqQEd8Alxr-LWpNPamz0E1geXMcUQ05nrvZ%7E5kEcR9nl-q0AdfjOf9A4pBlA__&Key-Pair-Id=KVTP0A1DKRTAX" -O pytorch_model-00003-of-00003.bin
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27tokenizer.model%3B+filename%3D%22tokenizer.model%22%3B&Expires=1685201630&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzY4LzA5LzY4MDk2MmM1ZWVkMTczYjFlMjQyYjE2OWRlMGI1NTQ1NGI3Y2Q3YWQwNGU1YzkzYjVkYjhkYjhkYWM5ZWM2OGMvOWU1NTZhZmQ0NDIxM2I2YmQxYmUyYjg1MGViYmJkOThmNTQ4MTQzN2E4MDIxYWZhZjU4ZWU3ZmIxODE4ZDM0Nz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODUyMDE2MzB9fX1dfQ__&Signature=urTbIUi9GmzR6F7iz9HYi%7EB-YVW-c3LmNYiQktwMMbC6fM1Vz7IcUwgZhpC97ZHF9jnL-q1x61C8i6Z8T5UZwoUvywe3RyI1c%7EJtaLZUx0U6PATYZBFEWzhCmSkkPQxjLcr%7E8jWVZiVujLcnu81-Eaj1B%7E7RoplWVoycTKobU3iIx4oC5ddbt9xW8rYSZ6y1Usnr5PTWLCjgYq8YCPRFVyRqXmMk%7EH5q7VHHzhj573IvbtprpWjlYBPg9HcEjbFJreTMWVonULpcCTLhaxIMMZ7hRHiPKo8V4gdYzfogCDA23KGMsRpPkIXgswMVqcEDY-bGmKKDZn5GzX3lIA%7EWGg__&Key-Pair-Id=KVTP0A1DKRTAX" -O tokenizer.model
cd ..

# generate alpaca ckpt
git clone git@github.com:tatsu-lab/stanford_alpaca.git
python stanford_alpaca/weight_diff.py recover --path_raw llama_hf_ckpt --path_diff alpaca-7b-wdiff --path_tuned alpaca_ckpt







# wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model -O ./tokenizer.model
# wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk -O ./tokenizer_checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/7B/consolidated.00.pth -O ./7B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/7B/params.json -O ./7B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/7B/checklist.chk -O ./7B/checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.00.pth -O ./13B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.01.pth -O ./13B/consolidated.01.pth
# wget https://agi.gpt4.org/llama/LLaMA/13B/params.json -O ./13B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/13B/checklist.chk -O ./13B/checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.00.pth -O ./30B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.01.pth -O ./30B/consolidated.01.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.02.pth -O ./30B/consolidated.02.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.03.pth -O ./30B/consolidated.03.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/params.json -O ./30B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/30B/checklist.chk -O ./30B/checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.00.pth -O ./65B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.01.pth -O ./65B/consolidated.01.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.02.pth -O ./65B/consolidated.02.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.03.pth -O ./65B/consolidated.03.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.04.pth -O ./65B/consolidated.04.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.05.pth -O ./65B/consolidated.05.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.06.pth -O ./65B/consolidated.06.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.07.pth -O ./65B/consolidated.07.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/params.json -O ./65B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/65B/checklist.chk -O ./65B/checklist.chk
