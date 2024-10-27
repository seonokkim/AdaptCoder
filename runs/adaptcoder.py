from executors import PyExecutor  # Importing execution engine for testing code.
from generators import PyGenerator, model_factory  # For generating code/debug prompts and creating model instances.
from typing import List  # Type hinting for specifying list type.
from multiprocessing import Pool  # For parallel processing.
from filelock import FileLock  # Locks file access to prevent race conditions.
from transformers import GPT2Tokenizer  # Imports GPT-2 tokenizer for token-based operations.
import random  # For selecting random failed tests for debugging.
import sys
from utils import *  # Import additional utility functions.

# Load GPT-2 tokenizer to track the number of tokens used during debugging for efficiency analysis.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_new_tests(failed_test, model, prompt, entry_point):
    """
    Generates edge cases based on failed tests to improve debugging robustness.
    Parameters:
        - failed_test: The test case that caused a failure.
        - model: The LLM used to generate edge cases.
        - prompt: The initial prompt to guide code generation.
        - entry_point: The function in code where testing begins.
    Returns:
        - new_tests: List of dynamically generated test cases targeting edge conditions.
    """
    new_tests = []
    response = model.generate(f"Generate edge cases for the test case: {failed_test}")
    for new_case in response:
        if entry_point in new_case and 'assert False' not in new_case:
            new_tests.append(new_case)
    return new_tests

def debug(i, item, log_path, model_name, num_items, pass_at_k, max_iters, port="", level="block"):
    """
    Core debugging function using AdaptCoder. Iteratively generates and tests code, refining it based on feedback.
    Parameters:
        - i: Index of the current item in dataset.
        - item: Data item containing code, test cases, etc.
        - log_path: Path to store debugging logs.
        - model_name: The model to be used for debugging.
        - num_items: Total number of items in dataset.
        - pass_at_k: Retry count for debugging each function.
        - max_iters: Maximum iterations for refining each function.
        - port, level: Additional params for model interaction.
    """
    exe = PyExecutor()  # Initialize the execution engine.
    gen = PyGenerator()  # Initialize the generator for creating debug messages.
    model = model_factory(model_name, port)  # Get specified LLM model instance.
    cur_pass = 0
    is_solved = False  # Track whether debugging has solved the function.
    implementations = []  # Store each iteration's implementation of the function.
    test_feedback = []  # Collect feedback on failed tests per iteration.
    token_nums = 0  # Track token count for efficient debugging.

    # Identify dataset type based on task ID for debug handling.
    dataset_type = item["task_id"].split("/")[0]
    
    while cur_pass < pass_at_k and not is_solved:
        # Filter test cases based on entry point, removing irrelevant ones.
        tests_i = item['given_tests']
        tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]

        # Initial attempt to generate a solution.
        cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], item["seed"], item["entry_point"])
        implementations.append(cur_func_impl)
        is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)  # Execute code to capture test results.
        test_feedback.append(failed_tests)

        # Exit if solution passes all tests in the first attempt.
        if is_passing:
            is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
            break
        
        # Iteratively refine the code for failed test cases.
        while cur_pass < pass_at_k and not is_solved:
            for _ in range(max_iters):
                debug_cur_func_impl = prepare_for_debugging(dataset_type, cur_func_impl, item)
                # Select a random failed test case for debugging.
                selected_test = random.choice(failed_tests) if failed_tests else None
                messages = gen.adaptcoder_debug(item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], model, "", dataset_type, level)
                cur_func_impl, cur_messages = gen.adaptcoder_debug_generate(item["prompt"], model, cur_func_impl, messages, selected_test, dataset_type)
                
                # Track tokens used during this debug step.
                token_nums += len(tokenizer.tokenize(cur_messages if isinstance(cur_messages, str) else "".join(cur_messages)))
                
                # Add updated implementation and re-test.
                implementations.append(cur_func_impl)
                is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(failed_tests)
                
                if is_passing:
                    is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_solved:
                        item["solution"] = cur_func_impl  # Save solution if passing.
                    break
            cur_pass += 1  # Increment debug pass counter.
            
            # Generate new edge cases for unresolved failures to refine further.
            if not is_solved:
                for failed_test in failed_tests:
                    item['given_tests'].extend(generate_new_tests(failed_test, model, item["prompt"], item["entry_point"]))
    
    # Save results, including test feedback and final solution details.
    item.update({
        "is_passing": is_passing,
        "is_solved": is_solved,
        "implementations": implementations,
        "test_feedback": test_feedback,
        "solution": cur_func_impl,
        "generated_test": tests_i,
        "debug_iter": cur_pass,
        "token_nums": token_nums
    })
    
    with FileLock(log_path + ".lock"):  # Lock log file to prevent race conditions in parallel runs.
        write_jsonl(log_path, [item], append=True)
    print(f'Completed {i+1}/{num_items}')

def run_adaptcoder(
    dataset: List[dict],
    model_name: str,
    max_iters: int,
    n_proc: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    seedfile: str = None,
    testfile: str = None,
    port: str = "",
    level: str = "block"
) -> None:
    """
    Executes AdaptCoder on a dataset in parallel, tracking success rates.
    Parameters:
        - dataset: Collection of data items (tasks) to debug.
        - model_name: Model for code generation/debugging.
        - max_iters: Max iterations for each debugging cycle.
        - n_proc: Number of parallel processes.
        - pass_at_k: Max attempts for each debugging pass.
        - log_path: Path to save results.
        - verbose: Toggle for detailed output.
        - seedfile, testfile, port, level: Optional parameters for dataset, testing, and model settings.
    """
    print("Number of proc:", n_proc)
    num_items = len(dataset)
    # Generate parameters for each item in dataset.
    args = [(i, item, log_path, model_name, num_items, pass_at_k, max_iters, port, level) for i, item in enumerate_resume(dataset, log_path, seedfile, testfile)]
    
    if n_proc == 1:
        for item in args:
            debug(*item)  # Run debug function sequentially.
    else:
        with Pool(n_proc) as pool:
            pool.starmap(debug, args)  # Parallel debugging.
    
    print("Accuracy:", count_solved(log_path))  # Output overall success rate after completion.
