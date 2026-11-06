import unittest
import sys
import types
from unittest.mock import patch

from services.task_errors import (
    SYSTEMIC_ERROR_CODES,
    TaskFailure,
    failure_stage,
    is_invalid_source_error,
    is_out_of_memory_error,
)
from services import task_errors
from utils import text_postprocess


class TaskErrorContractTests(unittest.TestCase):
    def test_systemic_error_codes_are_explicit_and_stable(self):
        self.assertEqual(SYSTEMIC_ERROR_CODES, frozenset({
            "model_download_failed", "model_unavailable", "dependency_missing",
            "device_unavailable", "device_error", "out_of_memory", "storage_unavailable",
        }))

    def test_task_failure_preserves_message_code_and_optional_stage(self):
        failure = TaskFailure("GPU allocation failed", "out_of_memory", stage="recognition")
        self.assertEqual(str(failure), "GPU allocation failed")
        self.assertEqual(failure.message, "GPU allocation failed")
        self.assertEqual(failure.code, "out_of_memory")
        self.assertEqual(failure_stage(failure), "recognition")
        self.assertEqual(failure_stage(RuntimeError("ordinary error")), "result")

    def test_oom_classifier_requires_an_allocation_failure_signature(self):
        self.assertTrue(is_out_of_memory_error("CUDA_ERROR_OUT_OF_MEMORY"))
        self.assertTrue(is_out_of_memory_error("vkAllocateMemory failed: device lost"))
        self.assertTrue(is_out_of_memory_error("failed to allocate buffer"))
        self.assertFalse(is_out_of_memory_error("ggml_vk_buffer initialized successfully; input decode failed"))

    def test_decode_failure_is_a_source_failure(self):
        message = "ggml_vk_buffer initialized successfully; input decode failed"
        self.assertTrue(is_invalid_source_error(message))
        self.assertFalse(is_out_of_memory_error(message))
        self.assertTrue(is_invalid_source_error("input decode failed: CUDA_ERROR_OUT_OF_MEMORY"))
        self.assertTrue(is_out_of_memory_error("input decode failed: CUDA_ERROR_OUT_OF_MEMORY"))

    def test_missing_punctuation_package_is_typed_dependency_missing(self):
        with patch.object(text_postprocess, "_PUNCTUATION_MODEL", None), \
                patch.dict(sys.modules, {"sbert_punc_case_ru": None}):
            with self.assertRaises(task_errors.TaskFailure) as raised:
                text_postprocess.load_punctuation_model()
        self.assertEqual(raised.exception.code, "dependency_missing")
        self.assertEqual(raised.exception.stage, "postprocess")

    def test_punctuation_model_load_failure_is_typed_model_unavailable(self):
        package = types.ModuleType("sbert_punc_case_ru")

        def fail_load():
            raise RuntimeError("bad model weights")

        package.SbertPuncCase = fail_load
        with patch.object(text_postprocess, "_PUNCTUATION_MODEL", None), \
                patch.dict(sys.modules, {"sbert_punc_case_ru": package}):
            with self.assertRaises(task_errors.TaskFailure) as raised:
                text_postprocess.load_punctuation_model()
        self.assertEqual(raised.exception.code, "model_unavailable")
        self.assertEqual(raised.exception.stage, "postprocess")


if __name__ == "__main__":
    unittest.main()
