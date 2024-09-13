
import asyncio

import yaml

from apropos.src.bench.bigcodebench.dags.plan_act import (
    code_problem_plan_execute_example,
)
from apropos.src.bench.bigcodebench.main import BigCodeBenchComplete_Benchmark
from apropos.src.bench.hendryks_math.dags.plan_act import (
    hendryks_math_plan_execute_example,
)
from apropos.src.bench.hendryks_math.main import HendryksMath_Benchmark

#from apropos.src.bench.bigcodebench.single_step_dag import bcb_plan_execute_example
from apropos.src.core.optimizers.miprov2p1.algorithm import MIPrO_V2p1_DAG

if __name__ == "__main__":
    with open('src/icl_bench/optimizers/mipro_v2p1/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)


    if config['problem_type'] == "CODING":
        benchmark = BigCodeBenchComplete_Benchmark()
        dag = code_problem_plan_execute_example(
            model_names=["gpt-4o-mini"] * 2
        )
    else:
        benchmark = HendryksMath_Benchmark()
        dag = hendryks_math_plan_execute_example(
            model_names=["gpt-4o-mini"] * 2
        )

    
    mipro_v2p1 = MIPrO_V2p1_DAG(
        student_program=dag,
        dataset_handler=benchmark,
        teacher_program=dag,
        cfg={
            "seed": 42,
            "n_optuna_trials": config['n_epochs'],
            "dev_size": config['dev_size'],
            "learnable_questions": {
                "max_n_to_obtain": 20,
                "max_n_to_sample": 100,
                "base_temp": 0.0,
                "k_for_pass_at_k": 5,
            },
        },
    )
    best_program = asyncio.run(mipro_v2p1.optimize_program())
    baseline_dag_scores = benchmark.score_dag_parsync(
        dag, split="test", n=config['test_size']
    )
    optimized_dag_scores = benchmark.score_dag_parsync(
        best_program, split="test", n=config['test_size']
    )
    print(
        "Baseline Program Performance: ",
        sum(baseline_dag_scores) / len(baseline_dag_scores),
    )
    print(
        "Optimized Program Performance: ",
        sum(optimized_dag_scores) / len(optimized_dag_scores),
    )
# E[Optimized] - E[Base]

# E[Optimized | optimizer] - E[Optimized | baseline LM]

# Results with gpt-4o-mini orchestrator

# gpt-4o-mini/gpt-4o-mini: 66-63 (v little compute)