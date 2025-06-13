import os
import signal
import logging
from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Any, Dict, List
from utils.model_loader import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import calculate_average_scores
import numpy as np

logger = logging.getLogger("evaluation")
np.random.seed(42)

class BaseEvaluator(ABC):
    """
    Every concrete evaluator must implement:
        load_data()      -> iterable
        evaluate_one(ex) -> dict with at least 'score' key
    """

    NAME: str = "base"  

    def __init__(self, args: Namespace):
        self.args = args

        self.cand = LLM(
            args.model_path,
            api_key=os.getenv("CAND_API_KEY"),
            base_url=os.getenv("CAND_BASE_URL"),
        )

        if self.args.judge_model:
            self.judge = LLM(
                args.judge_model,
                api_key=os.getenv("JUDGE_API_KEY"),
                base_url=os.getenv("JUDGE_BASE_URL"),
            )

    @classmethod
    @abstractmethod
    def add_args(cls, p: ArgumentParser): ...
    @abstractmethod
    def load_data(self): ...
    @abstractmethod
    def evaluate_one(self, example) -> Dict[str, Any]: ...


    def run(self):
        data = list(self.load_data())
        results = []

        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor = ThreadPoolExecutor(max_workers=self.args.workers)
        signal.signal(signal.SIGINT, original_sigint_handler)

        try:
            futures = [executor.submit(self.evaluate_one, ex) for ex in data]

            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                _,scores_string=calculate_average_scores(results)
                logger.info(
                    f"{self.NAME}: {len(results)}/{len(data)} " f"scores={scores_string}"
                )

        except KeyboardInterrupt:
            logger.warning("Interrupted by user – cancelling leftover jobs…")
            for f in futures:
                f.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

            if results:
                self._dump(results)

        return results

    def _dump(self, res):
        if res: 
            with open(f"thursday_2_results_{self.NAME}.txt", "a") as f:
                _,scores_string = calculate_average_scores(res)
                args_dict = vars(self.args)  
                args_string = ', '.join([f"{key}={value}" for key, value in args_dict.items()]) 
                f.write(f"{self.args.model_path}  scores={scores_string}  args={{ {args_string} }}\n")



    def bootstrap_run(self, n_samples: int = 120, n_iterations: int = 5):
        print('bootstrap run')
        """
        Run bootstrap evaluation with replacement
        n_samples: number of samples to use in each iteration
        n_iterations: number of bootstrap iterations
        """
        all_data = list(self.load_data())
        all_bootstrap_results = []
        all_scores = []

        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor = ThreadPoolExecutor(max_workers=self.args.workers)
        signal.signal(signal.SIGINT, original_sigint_handler)

        try:
            for iteration in range(n_iterations):
                bootstrap_indices = np.random.choice(
                    len(all_data), size=n_samples, replace=False
                )
                bootstrap_data = [all_data[i] for i in bootstrap_indices]
                
                results = []
                futures = [executor.submit(self.evaluate_one, ex) for ex in bootstrap_data]
                
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:  
                        results.append(res)
                        _ ,scores_string=calculate_average_scores(results)
                        logger.info(
                            f"{self.NAME}: {len(results)}/{n_samples} " f"scores={scores_string}"
                        )
                
                if results:
                    scores, _  = calculate_average_scores(results)
                    all_scores.append(scores)
                    all_bootstrap_results.append(results)
                    
                    mean_scores = {}
                    std_scores = {}
                    for metric in scores.keys():
                        values = [s[metric] for s in all_scores if metric in s]
                        mean_scores[metric] = np.mean(values)
                        std_scores[metric] = np.std(values)
                    
                    logger.info(
                        f"\nBootstrap iteration {iteration + 1}/{n_iterations}\n"
                        f"Current means: {mean_scores}\n"
                        f"Current stds: {std_scores}\n"
                    )

        except KeyboardInterrupt:
            logger.warning("Interrupted by user – cancelling leftover jobs…")
            for f in futures:
                f.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            
            if all_scores:
                final_stats = self._calculate_bootstrap_statistics(all_scores)
                self._dump_bootstrap_results(final_stats)
            
        return all_bootstrap_results, final_stats

    def _calculate_bootstrap_statistics(self, all_scores: List[Dict[str, float]]):
        """Calculate mean and standard deviation for all metrics"""
        stats = {}
        for metric in all_scores[0].keys(): 
            values = [s[metric] for s in all_scores if metric in s]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),  
                'ci_upper': np.percentile(values, 97.5)
            }
        return stats

    def _dump_bootstrap_results(self, stats: Dict[str, Dict[str, float]]):
        """Dump bootstrap results to file"""
        with open(f"bootstrap_results_{self.NAME}.txt", "a") as f:
            f.write(f"\nModel: {self.args.model_path}\n")
            f.write("Bootstrap Statistics:\n")
            for metric, values in stats.items():
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {values['mean']:.4f}\n")
                f.write(f"  Std:  {values['std']:.4f}\n")
                f.write(f"  95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]\n")
            f.write("\nArguments:\n")
            args_dict = vars(self.args)
            for key, value in args_dict.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "="*50 + "\n")
