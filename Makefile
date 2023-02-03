setup:
	[ -d ./venv ] || python3 -m venv ./venv --upgrade-deps
	./venv/bin/pip install -r requirements.txt

check:
	CUDA_VISIBLE_DEVICES= ./venv/bin/python test.py
	echo PASS/FAIL TO BE DECIDED BY YOU BASED ON OUTPUT PRINTED ABOVE
	echo IGNORE EXIT CODE
	exit 1
