PYTHON ?= .venv/bin/python
LB_DIR = src/1D-Laplace-Beltrami
LBU_DIR = src/1D-Laplace-Beltrami-Unordered

.PHONY: run laplace-beltrami laplace-unordered stop
.DEFAULT_GOAL := run

run: laplace-beltrami

laplace-beltrami: PORT = 8000
laplace-beltrami: stop
	$(PYTHON) $(LB_DIR)/arc_length_param.py

laplace-unordered: PORT = 8001
laplace-unordered: stop
	$(PYTHON) $(LBU_DIR)/unordered_param.py

stop:
	-@kill $$(lsof -tiTCP:$(PORT) -sTCP:LISTEN) 2>/dev/null
	@true
