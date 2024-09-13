# test gpt-4o => o1 => gpt-4o-mini sandwich
import asyncio

import networkx as nx

from apropos.src.core.programs.convenience_functions.dag_constructors import (
    build_dag_program,
)
from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    Topic,
    UserMessage,
)


def math_orion_sandwich(model_names=["gpt-4o-mini", "o1-mini", "gpt-4o-mini"]):
    simplify = PromptTemplate(
        name="Simplify Problem",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI assistant tasked with simplifying complex mathematics problems.",
                        "$MAIN_INSTRUCTIONS": "You will be given a mathematics problem statement. Your task is to identify the core problem and reframe it in the simplest possible terms for a logical reasoner to solve.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Simplify the given mathematics problem to its core essence. Focus on the key elements that a logical reasoner needs to solve the crux of the problem."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Math Problem\n<<<MATHEMATICS_QUESTION>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Please provide the simplified version of this problem, focusing on the core elements needed to solve it:",
                    },
                    input_fields=["<<<MATHEMATICS_QUESTION>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    
    solve = PromptTemplate(
        name="Solve Core Problem",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI logical reasoner with expertise in solving mathematical problems.",
                        "$MAIN_INSTRUCTIONS": "You will be given a simplified version of a mathematics problem. Your task is to solve the core problem using logical reasoning.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Solve the simplified math problem using logical reasoning. Provide a clear, step-by-step solution."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Simplified Problem\n<<<SIMPLIFIED_PROBLEM>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Please solve this simplified problem, showing your reasoning:"
                    },
                    input_fields=["<<<SIMPLIFIED_PROBLEM>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    
    present = PromptTemplate(
        name="Present Final Solution",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI assistant tasked with presenting mathematical solutions clearly and comprehensively.",
                        "$MAIN_INSTRUCTIONS": "You will be given the original problem, its simplified version, and the solution to the simplified problem. Your task is to present a complete and coherent final answer.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Present a comprehensive final answer that addresses the original problem, incorporating the simplified problem and its solution."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Original Problem\n<<<MATHEMATICS_QUESTION>>>\n\n# Simplified Problem\n<<<SIMPLIFIED_PROBLEM>>>\n\n# Solution\n<<<SOLUTION>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Please present the final comprehensive answer to the original problem:",
                    },
                    input_fields=["<<<MATHEMATICS_QUESTION>>>", "<<<SIMPLIFIED_PROBLEM>>>", "<<<SOLUTION>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    
    prompts = {
        "Simplify Problem": simplify,
        "Solve Core Problem": solve,
        "Present Final Solution": present
    }
    
    name_dag = nx.DiGraph()
    name_dag.add_edge("DAG_INPUT", "Simplify Problem")
    name_dag.add_edge("DAG_INPUT", "Present Final Solution", attribute_name="<<<MATHEMATICS_QUESTION>>>")
    name_dag.add_edge("Simplify Problem", "Solve Core Problem", attribute_name="<<<SIMPLIFIED_PROBLEM>>>")
    name_dag.add_edge("Simplify Problem", "Present Final Solution", attribute_name="<<<SIMPLIFIED_PROBLEM>>>")
    name_dag.add_edge("Solve Core Problem", "Present Final Solution", attribute_name="<<<SOLUTION>>>")
    name_dag.add_edge("Present Final Solution", "DAG_OUTPUT", attribute_name="<<<FINAL_ANSWER>>>")

    model_configs = {
        "Simplify Problem": {"model_name": model_names[0], "temperature": 0.0},
        "Solve Core Problem": {"model_name": model_names[1], "temperature": 1},
        "Present Final Solution": {"model_name": model_names[2], "temperature": 0.0}
    }

    math_orion_sandwich_dag = asyncio.run(
        build_dag_program(
            prompts=prompts,
            name_dag=name_dag,
            model_configs=model_configs,
            dag_input_names=["<<<MATHEMATICS_QUESTION>>>"],
            dag_input_aliases={"question": "<<<MATHEMATICS_QUESTION>>>"},
            dag_output_aliases={"<<<FINAL_ANSWER>>>": "answer"},
        )
    )
    return math_orion_sandwich_dag


def coding_orion_sandwich(model_names=["gpt-4o-mini", "o1-mini", "gpt-4o-mini"]):
    analyze = PromptTemplate(
        name="Analyze Problem",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI assistant tasked with analyzing complex coding problems.",
                        "$MAIN_INSTRUCTIONS": "You will be given a coding problem statement. Your task is to identify the core requirements and break down the problem into key components.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Analyze the given coding problem to its core components. Focus on identifying key algorithms, data structures, and computational steps needed to solve the problem."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Coding Problem\n<<<CODING_QUESTION>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Please provide a concise analysis of this problem, focusing on the core components needed to solve it:",
                    },
                    input_fields=["<<<CODING_QUESTION>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    
    implement = PromptTemplate(
        name="Implement Core Solution",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI coding expert with proficiency in implementing efficient solutions.",
                        "$MAIN_INSTRUCTIONS": "You will be given an analysis of a coding problem. Your task is to implement the core solution based on this analysis.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Implement the core solution to the coding problem based on the provided analysis. Focus on efficiency and correctness."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Problem Analysis\n<<<PROBLEM_ANALYSIS>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Please implement the core solution based on this analysis:"
                    },
                    input_fields=["<<<PROBLEM_ANALYSIS>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    
    refine = PromptTemplate(
        name="Refine and Present Solution",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="premise",
                    topic_template="# Premise\n$INTRODUCTION\n$MAIN_INSTRUCTIONS",
                    instructions_fields={
                        "$INTRODUCTION": "You are an AI assistant tasked with refining and presenting coding solutions comprehensively.",
                        "$MAIN_INSTRUCTIONS": "You will be given the original problem, its analysis, and the core implementation. Your task is to refine the solution and present a complete, well-documented answer.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="objective",
                    topic_template="# Objective\n$OBJECTIVE",
                    instructions_fields={
                        "$OBJECTIVE": "Refine the implemented solution and present a comprehensive final answer that addresses the original problem, including proper documentation and error handling."
                    },
                    input_fields=[],
                )
            ],
            constraints=[],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="user",
                    topic_template="# Original Problem\n<<<CODING_QUESTION>>>\n\n# Problem Analysis\n<<<PROBLEM_ANALYSIS>>>\n\n# Core Implementation\n<<<CORE_IMPLEMENTATION>>>\n\n $ENJOINDER",
                    instructions_fields={
                        "$ENJOINDER": "Please refine and present the final comprehensive solution to the original problem:",
                    },
                    input_fields=["<<<CODING_QUESTION>>>", "<<<PROBLEM_ANALYSIS>>>", "<<<CORE_IMPLEMENTATION>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    
    prompts = {
        "Analyze Problem": analyze,
        "Implement Core Solution": implement,
        "Refine and Present Solution": refine
    }
    
    name_dag = nx.DiGraph()
    name_dag.add_edge("DAG_INPUT", "Analyze Problem")
    name_dag.add_edge("DAG_INPUT", "Refine and Present Solution", attribute_name="<<<CODING_QUESTION>>>")
    name_dag.add_edge("Analyze Problem", "Implement Core Solution", attribute_name="<<<PROBLEM_ANALYSIS>>>")
    name_dag.add_edge("Analyze Problem", "Refine and Present Solution", attribute_name="<<<PROBLEM_ANALYSIS>>>")
    name_dag.add_edge("Implement Core Solution", "Refine and Present Solution", attribute_name="<<<CORE_IMPLEMENTATION>>>")
    name_dag.add_edge("Refine and Present Solution", "DAG_OUTPUT", attribute_name="<<<FINAL_SOLUTION>>>")

    model_configs = {
        "Analyze Problem": {"model_name": model_names[0], "temperature": 0.0},
        "Implement Core Solution": {"model_name": model_names[1], "temperature": 1},
        "Refine and Present Solution": {"model_name": model_names[2], "temperature": 0.0}
    }

    coding_orion_sandwich_dag = asyncio.run(
        build_dag_program(
            prompts=prompts,
            name_dag=name_dag,
            model_configs=model_configs,
            dag_input_names=["<<<CODING_QUESTION>>>"],
            dag_input_aliases={"question": "<<<CODING_QUESTION>>>"},
            dag_output_aliases={"<<<FINAL_SOLUTION>>>": "answer"},
        )
    )
    return coding_orion_sandwich_dag