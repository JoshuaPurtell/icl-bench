from apropos.src.bench.bigcodebench.main import BigCodeBenchComplete_Benchmark
from apropos.src.bench.bigcodebench.dags.single_step_dag import code_problem_single_step
from apropos.src.core.optimizers.baselines.bffsrs import BreadthFirstRandomSearch_DAG
from apropos.src.bench.hendryks_math.dags.single_step import hendryks_math_single_step_example
from apropos.src.bench.hendryks_math.main import HendryksMath_Benchmark
import asyncio
import yaml

if __name__ == "__main__":
    with open('src/icl_bench/optimizers/random_search/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    if config['problem_type'] == "MATH":
        benchmark = HendryksMath_Benchmark()
        dag = hendryks_math_single_step_example()
    else:
        benchmark = BigCodeBenchComplete_Benchmark()
        dag = code_problem_single_step(
            model_name="gpt-4o-mini"
        )
    bffsrs_config = {
        "optimization": {
            "combination_size": 3,
            "n_iterations": config['n_epochs'],
            "validation_size": config['dev_size'],
            "test_size": config['test_size'],
            "program_search_parallelization_factor": 5,
        },
        "bootstrapping": {
            "patches": ["A", "B"],
            "n_questions": 100,
        },
        "verbose": True,
    }
    bffsrs = BreadthFirstRandomSearch_DAG(
        student_program=dag,
        teacher_program=dag,
        dataset_handler=benchmark,
        cfg=bffsrs_config,
    )
    print("Optimizing demonstrations...")
    optimized_dag = asyncio.run(bffsrs.optimize_demonstrations())

    print("Evaluating optimized program...")
    optimized_program_scores, _ = benchmark.score_dag_parsync(
        optimized_dag, n=config['test_size'], verbose=True, split="test", patches=["A", "B"]
    )
    print("Optimized program scores:", sum(optimized_program_scores)/len(optimized_program_scores))
    print("Evaluating baseline program...")
    baseline_program_scores, _ = benchmark.score_dag_parsync(
        dag, n=config['test_size'], verbose=True, split="test", patches=["A", "B"]
    )
    print("Baseline program scores:", sum(baseline_program_scores)/len(baseline_program_scores))

