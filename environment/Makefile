.PHONY: environment remove-env test

MODULE=gnn
ENVIRONMENT=gnn_workspace

environment: 
	conda env create -f environment.yml

remove-env:
	conda remove --name $(ENVIRONMENT) --all --yes

test:
	pytest -v
