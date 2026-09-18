PYTHON ?= .venv/bin/python
PORT ?= 8000

.PHONY: run stop
.DEFAULT_GOAL := run

run: stop
	$(PYTHON) src/arc_length_param.py

stop:
	-@kill $$(lsof -tiTCP:$(PORT) -sTCP:LISTEN) 2>/dev/null
	@true
