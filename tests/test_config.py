from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import dotenv
import finance_rag.config as config
from finance_rag.config import env_bool, load_project_env, resolve_chunk_tokenizer


class ProjectEnvironmentTests(unittest.TestCase):
    def test_project_env_loads_values_without_overwriting_process_environment(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text(
                "ENABLE_SMART_CHUNKER=true\nRERANKER_DEVICE=cuda\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"RERANKER_DEVICE": "cpu"},
                clear=True,
            ):
                load_project_env(root)
                self.assertEqual(os.environ["ENABLE_SMART_CHUNKER"], "true")
                self.assertEqual(os.environ["RERANKER_DEVICE"], "cpu")


class BooleanConfigTests(unittest.TestCase):
    def test_env_bool_accepts_documented_true_and_false_values(self):
        cases = {
            "true": True,
            " 1 ": True,
            "YES": True,
            "On": True,
            "false": False,
            " 0 ": False,
            "NO": False,
            "Off": False,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"FEATURE_FLAG": raw}, clear=True
            ):
                self.assertIs(env_bool("FEATURE_FLAG", not expected), expected)

    def test_env_bool_uses_default_only_when_variable_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(env_bool("FEATURE_FLAG", True))
            self.assertFalse(env_bool("FEATURE_FLAG", False))

    def test_env_bool_rejects_invalid_nonempty_value_with_name_and_value(self):
        with patch.dict(os.environ, {"FEATURE_FLAG": "enabled"}, clear=True):
            with self.assertRaisesRegex(ValueError, "FEATURE_FLAG.*enabled"):
                env_bool("FEATURE_FLAG", False)


class ChunkTokenizerTests(unittest.TestCase):
    def test_chinese_switch_forces_bge_over_explicit_tokenizer(self):
        self.assertEqual(
            resolve_chunk_tokenizer(True, "custom/tokenizer"),
            "BAAI/bge-large-zh-v1.5",
        )

    def test_disabled_chinese_switch_uses_explicit_tokenizer(self):
        self.assertEqual(
            resolve_chunk_tokenizer(False, "custom/tokenizer"),
            "custom/tokenizer",
        )

    def test_disabled_chinese_switch_defaults_to_minilm(self):
        self.assertEqual(
            resolve_chunk_tokenizer(False, None),
            "sentence-transformers/all-MiniLM-L6-v2",
        )


class OptimizationSwitchTests(unittest.TestCase):
    def test_every_boolean_switch_uses_strict_parser(self):
        switch_names = (
            "USE_ZH_TOKENIZER",
            "ENABLE_RERANKER",
            "CHAT_ENABLE_QUERY_REWRITE",
            "ENABLE_SMART_CHUNKER",
            "ENABLE_MULTI_STAGE_RETRIEVAL",
            "ENABLE_METADATA_FILTER",
            "ENABLE_FINANCIAL_EXPERT_PROMPT",
            "ENABLE_CITATION_VALIDATION",
        )
        for name in switch_names:
            with self.subTest(name=name), patch.dict(
                os.environ, {name: "invalid"}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, name):
                    env_bool(name, False)

    def test_invalid_switch_values_fail_during_production_config_initialization(self):
        switch_names = (
            "USE_ZH_TOKENIZER",
            "ENABLE_RERANKER",
            "CHAT_ENABLE_QUERY_REWRITE",
            "ENABLE_SMART_CHUNKER",
            "ENABLE_MULTI_STAGE_RETRIEVAL",
            "ENABLE_METADATA_FILTER",
            "ENABLE_FINANCIAL_EXPERT_PROMPT",
            "ENABLE_CITATION_VALIDATION",
        )
        try:
            for name in switch_names:
                with self.subTest(name=name), patch.dict(
                    os.environ,
                    {"DEEPSEEK_API_KEY": "", name: "invalid"},
                    clear=True,
                ), patch("dotenv.load_dotenv"):
                    with self.assertRaisesRegex(ValueError, name):
                        importlib.reload(config)
        finally:
            config.load_dotenv = dotenv.load_dotenv
