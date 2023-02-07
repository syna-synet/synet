setup:
	[ -d ./venv ] || python3 -m venv ./venv --upgrade-deps
	./venv/bin/pip install -r requirements.txt

check:
	TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES= ./venv/bin/python test.py
