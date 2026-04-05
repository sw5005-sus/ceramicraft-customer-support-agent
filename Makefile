.PHONY: gen setup test

gen:
	buf generate
	perl -pi -e 's/^import (.*_pb2.*)/from . import $$1/' src/ceramicraft_customer_support_agent/pb/cs_agent_pb2_grpc.py
	touch src/ceramicraft_customer_support_agent/pb/__init__.py

setup:
	uv sync --dev

test:
	uv run pytest -v
