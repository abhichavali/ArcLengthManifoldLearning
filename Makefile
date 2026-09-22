PYTHON ?= .venv/bin/python
PORT ?= 8000
LB_DIR = src/1D-Laplace-Beltrami

.PHONY: run laplace-beltrami stop
.DEFAULT_GOAL := run

run: laplace-beltrami

laplace-beltrami: stop
	$(PYTHON) $(LB_DIR)/arc_length_param.py

stop:
	-@kill $$(lsof -tiTCP:$(PORT) -sTCP:LISTEN) 2>/dev/null
	@true
