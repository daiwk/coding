
pwd_dir=`pwd`

# refer to https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html#install-verl-upstream

# Create a virtual environment
python3 -m venv ~/.python/verl_env

# Activate the virtual environment
source ~/.python/verl_env/bin/activate

# Install uv
python3 -m pip install uv

cd ~
rm -rf verl
git clone https://github.com/volcengine/verl.git
cd verl

# Install verl
python3 -m uv pip install .
cd $pwd_dir
# 这个是把原来requirements文件里的flash-attn删了
python3 -m uv pip install -r ./requirements-sglang-nofa2.txt --no-cache-dir

python3 -m uv pip install "protobuf==3.20.3"
python3 -m uv pip install "transformers<=4.57.6"

python -m pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3

python -m pip install cachetools

cd ~/verl
cd ./examples/data_preprocess
python3 gsm8k.py --local_save_dir ~/data/gsm8k

