all:
	cat Makefile

test:
	python3 train.py data --epochs 1 --logdir resnet18

demo:
	./scripts/baseline.sh


clean:
	rm -rf train_log/

