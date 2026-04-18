"""Configuration for DeepEval LLM evaluation tests.

These tests require OPENAI_API_KEY. They are excluded from normal pytest
runs via --ignore=tests/deepeval in CI workflows (test.yml, sonar.yml).
The dedicated DeepEval workflow provides the key.
"""
