from executors import PyExecutor
from generators import PyGenerator, model_factory
from typing import List
from multiprocessing import Pool
from filelock import FileLock
from transformers import GPT2Tokenizer
import random
import sys
from utils import *

# Load GPT-2 tokenizer for tokenization metrics
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_new_tests(failed_test, model, prompt, entry_point):
    """Generate edge cases dynamically based on failed test cases to improve robustness."""
    new_tests = []
    response = model.generate(f"Generate edge cases for the test case: {failed_test}")
    for new_case in response:
        if entry_point in new_case and 'assert False' not in new_case:
            new_tests.append(new_case)
    return new_tests

def debug(i, item, log_path, model_name, num_items, pass_at_k, max_iters, port="", level="block"):
    """Core debugging function that iteratively improves code using AdaptCoder."""
    exe = PyExecutor()
    gen = PyGenerator()
    model = model_factory(model_name, port)
    cur_pass = 0
    is_solved = False
    implementations = []
    test_feedback = []
    token_nums = 0
    
    dataset_type = item["task_id"].split("/")[0]
    
    while cur_pass < pass_at_k and not is_solved:
        tests_i = item['given_tests']
        tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]

        # First attempt to solve using the base implementation
        cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], item["seed"], item["entry_point"])
        implementations.append(cur_func_impl)
        is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
        test_feedback.append(failed_tests)

        if is_passing:
            is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
            break
        
        # Iteratively refine solution until solved or attempts exhausted
        while cur_pass < pass_at_k and not is_solved:
            for _ in range(max_iters):
                debug_cur_func_impl = prepare_for_debugging(dataset_type, cur_func_impl, item)
                selected_test = random.choice(failed_tests) if failed_tests else None
                messages = gen.AdaptCoder_debug(item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], model, "", dataset_type, level)
                cur_func_impl, cur_messages = gen.AdaptCoder_generate(item["prompt"], model, cur_func_impl, messages, selected_test, dataset_type)
                
                # Track tokens used for debugging
                token_nums += len(tokenizer.tokenize(cur_messages if isinstance(cur_messages, str) else "".join(cur_messages)))
                
                implementations.append(cur_func_impl)
                is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(failed_tests)
                
                if is_passing:
                    is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_solved:
                        item["solution"] = cur_func_impl
                    break
            cur_pass += 1
            
            # Add new edge cases for unresolved failed tests to focus debugging
            if not is_solved:
                for failed_test in failed_tests:
                    item['given_tests'].extend(generate_new_tests(failed_test, model, item["prompt"], item["entry_point"]))
    
    # Log and save results
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
    
    with FileLock(log_path + ".lock"):
        write_jsonl(log_path, [item], append=True)
    print(f'Completed {i+1}/{num_items}')

def run_AdaptCoder(
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
    """Executes AdaptCoder over the dataset in parallel, tracking overall success rates."""
    print("Number of proc:", n_proc)
    num_items = len(dataset)
    args = [(i, item, log_path, model_name, num_items, pass_at_k, max_iters, port, level) for i, item in enumerate_resume(dataset, log_path, seedfile, testfile)]
    
    if n_proc == 1:
        for item in args:
            debug(*item)
    else:
        with Pool(n_proc) as pool:
            pool.starmap(debug, args)
    
    print("Accuracy:", count_solved(log_path))