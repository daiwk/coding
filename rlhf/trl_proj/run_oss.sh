
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
pip3 install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" trackio

export NO_PROXY="localhost, 127.0.0.1, ::1"
python3 finetune_oss.py
