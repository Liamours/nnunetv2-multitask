import unittest

from nnunetv2.tests.check_multitask_model import run_multitask_model_checker


class TestMultiTaskModelChecker(unittest.TestCase):
    def test_model_checker(self):
        report = run_multitask_model_checker()
        self.assertIn("dual_head", report["variants"])
        self.assertIn("dual_decoder", report["variants"])
        self.assertTrue(report["trainer"]["loss_key_present"])
        self.assertTrue(report["trainer"]["validation_loss_key_present"])
        self.assertEqual(set(report["inference"]["prediction_tasks"].keys()), {"task1", "task2"})


if __name__ == "__main__":
    unittest.main()
