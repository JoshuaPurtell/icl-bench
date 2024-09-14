# ICL-Bench

Evaluating Language Models' Ability to Learn In Context

## Optimizers

### Random Search + Single Step BCB/MATH
Define Delta as (E[Optimized] - E[Baseline])/ SD(Optimized). Deltas below 1 probably aren't meaningful, Deltas below 2 may not be meaningful.

| Student Model Name | Teacher Model Name | N Demos | Task | DAG | Delta | Score Mean | Score Std | Baseline Score | Train Compute | Test Cost |
|---------------------|---------------------|---------|------|-----|-------|------------|-----------|----------------|---------------|-----------|
| gpt-4o-mini         | gpt-4o-2024-08-06        | 4       | MATH | TRIVIAL | 0.6238 | 0.661      | 0.009     | 0.655          | ???           | ????      |
| gpt-4o-mini         | gpt-4o-mini         | 4       | MATH | TRIVIAL | -0.5656 | 0.645     | 0.017     | 0.655          | ???           | ????      |

### MIPRO v2.1 + Plan-Act/Reformulate-Reason-Return MATH/BCB - Student

| Model Name | Task | DAG      | Score Mean | Score Std | Train Compute | Test Cost |
|------------|------|----------|------------|-----------|---------------|-----------|
| gpt-4o-mini| MATH | PLAN_ACT | ???        | ???       | ???           | ???       |

### MIPRO v2.1 + Plan-Act MATH/BCB - Orchestrator

## Mean Few-Shot Demo Improvement

## Mean Trajectory Demo Improvement


We used meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo for L3.1-8b-instruct*

Todos 
- add Orion sandwich
- get scores