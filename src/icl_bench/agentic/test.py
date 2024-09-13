import smallbench
import apropos
from apropos.src.core.lms.helpers import LLM
from smallbench.baselines.agents.experimental.trajectory_guided import FewshotTrajectoryDemoReActLanguageAgent
from smallbench.benchmarks.bcb_a.test import get_contexts_extremely_hacky_please_fix
import asyncio
from typing import List, Dict, Any
import time
import random
from smallbench.benchmarks.bcb_a.bench import BCB_AgentBenchmark

async def score_agent_async(
        trajectory_demos: List[str],
        contexts_for_agent: List[Dict[str, Any]],
        model_name: str,
        indices: List[int] = [i for i in range(0,20)]
):
    agent = FewshotTrajectoryDemoReActLanguageAgent(
        trajectory_demos=trajectory_demos,
        lm=LLM(model_name), contexts=contexts_for_agent, multi_threaded=False
    )
    agent_benchmark = BCB_AgentBenchmark(backend="modal")
    agent_performances, agent_cost, agents = await agent_benchmark.score_agent_async(agent, split="train", indices=indices, verbose=False)
    overall_performance = sum(agent_performances) / len(agent_performances)
    successful_trajectories = [str(agent.react_history) for agent_performance, agent in zip(agent_performances, agents) if agent_performance == 1]
    return overall_performance, successful_trajectories


async def score_agent_with_trajectories(k_trajectories, n_samples, trajectory_demos, contexts_for_agent, model_name):
    random.seed(42)
    agent_scores = []
    for _ in range(n_samples):
        trajectory_demos = random.sample(trajectory_demos, k_trajectories)
        agent_score, _ = await score_agent_async(trajectory_demos, contexts_for_agent, model_name)
        agent_scores.append(agent_score)
    return sum(agent_scores) / n_samples

async def run_trajectory_demo_experiment(ks: List[int], n_samples: int, contexts_for_agent, model_name):
    baseline_performance, trajectories = await score_agent_async([], contexts_for_agent, model_name)
    print(f"Baseline Performance: {baseline_performance}")
    for k in ks:
        performance = await score_agent_with_trajectories(k, n_samples, trajectories, contexts_for_agent, model_name)
        print(f"K: {k}, Performance: {performance}")

# E[Performance with trajectory demos] - E[Performance zero-shot]
if __name__ == "__main__":
    contexts_for_agent = get_contexts_extremely_hacky_please_fix()
    asyncio.run(run_trajectory_demo_experiment([1, 2, 3, 4, 5], 10, contexts_for_agent, "gpt-4o-mini"))
