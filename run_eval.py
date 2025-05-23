import time
import logging
import importlib
from evaluators import BaseEvaluator
from argparse import ArgumentParser

logging.basicConfig(
    level=logging.INFO,  # Set to INFO level
    format="%(asctime)s %(levelname)s:%(message)s",  # Format for log messages
    handlers=[logging.StreamHandler()],
)  # Log to standard output

ALL_EVALUATORS = {
    "cypher": "evaluators.cypher.CypherEvaluator",
    "qa": "evaluators.qa.QAEvaluator",
    "bird": "evaluators.bird.BirdEvaluator",
    "clinton": "evaluators.clinton.ClintonEvaluator",
    "crt_qa": "evaluators.crt_qa.CRTQAEvaluator",
    "mimic": "evaluators.mimic.MimicEvaluator",
    "tablebench": "evaluators.tablebench.TableBenchEvaluator",
}


def dynamic_import(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, cls_name)


def main() -> None:
    root = ArgumentParser("Universal evaluation harness")

    root.add_argument(
        "--task", required=True, choices=ALL_EVALUATORS, help="Which benchmark to run"
    )

    root.add_argument(
        "--model_path", required=True, help="OpenAI compatible model to evaluate"
    )
    root.add_argument("--judge_model", help="OpenAI compatible model to use as judge")
    root.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel samples to evaluate.",
    )
    root.add_argument(
        "--base_dir",
        type=str,
        default="/dcai/projects02/data/cxrh/eval_data/eval_data",
        help="The base directory to load data from.",
    )

    # parse
    global_args, remaining_argv = root.parse_known_args()
    EClass: BaseEvaluator = dynamic_import(ALL_EVALUATORS[global_args.task])
    EClass.add_args(root)
    args = root.parse_args()

    t0 = time.time()
    evaluator: BaseEvaluator = EClass(args)
    results = evaluator.run()
    acc = sum(r["score"] for r in results) / len(results)
    logging.info(
        f"Finished evaluation – {acc*100:.2f}% " f"(elapsed {time.time()-t0:.1f}s)"
    )


if __name__ == "__main__":
    main()
