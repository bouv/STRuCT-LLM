import time
import logging
import importlib
from evaluators import BaseEvaluator
from argparse import ArgumentParser
from utils.helpers import calculate_average_scores

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s %(levelname)s:%(message)s", 
    handlers=[logging.StreamHandler()],
) 

ALL_EVALUATORS = {
    "cypher": "evaluators.cypher.CypherEvaluator",
    "qa": "evaluators.qa.QAEvaluator",
    "bird": "evaluators.bird.BirdEvaluator",
    "crt_qa": "evaluators.crt_qa.CRTQAEvaluator",
    "mimic": "evaluators.mimic.MimicEvaluator",
    "tablebench": "evaluators.tablebench.TableBenchEvaluator",
    "spider": "evaluators.spider.SpiderEvaluator",
    "SM3": "evaluators.SM3.SM3Evaluator",
    "SM3_examples":"evaluators.SM3_examples.SM3Evaluator",
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
        default=30,
        help="Number of parallel samples to evaluate.",
    )
    root.add_argument(
        "--base_dir",
        type=str,
        default="data",
        help="The base directory to load data from.",
    )
    root.add_argument(
        "--bootstrap",
        type=int,
        default=1,
        help="If 1 use bootstrap sampling instead of full dataset evaluation"
    )
    root.add_argument(
        "--bootstrap_samples",
        type=int,
        default=120,
        help="Number of samples to use in each bootstrap iteration"
    )
    root.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5,
        help="Number of bootstrap iterations to perform"
    )

    

    global_args, remaining_argv = root.parse_known_args()
    EClass: BaseEvaluator = dynamic_import(ALL_EVALUATORS[global_args.task])
    BaseEvaluator.add_args(root) 
    EClass.add_args(root)
    args = root.parse_args()

    t0 = time.time()
    evaluator: BaseEvaluator = EClass(args)

    if args.bootstrap==1:
        results, stats = evaluator.bootstrap_run(
            n_samples=args.bootstrap_samples,
            n_iterations=args.bootstrap_iterations
        )
        
        logging.info(
            f"\nFinished bootstrap evaluation:"
            f"\nFinal Statistics:"
        )
        for metric, values in stats.items():
            logging.info(
                f"\n{metric}:"
                f"\n  Mean: {values['mean']:.4f}"
                f"\n  Std:  {values['std']:.4f}"
                f"\n  95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]"
            )
    else:
        results = evaluator.run()

        if results:
            scores_display=calculate_average_scores(results)

            logging.info(
                f"Finished evaluation:"
                f"{scores_display}\n"
                f"Elapsed time: {time.time()-t0:.1f}s"
                )
    logging.info(f"Elapsed time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
