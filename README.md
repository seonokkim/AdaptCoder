# AdaptCoder: Adaptive Code Generation and Efficient Debugging Agents with LLM

![AdaptCoder Diagram](https://github.com/seonokkim/AdaptCoder/blob/main/figure/AdaptCoder.jpg)


**AdaptCoder** is a project-in-progress aimed at advancing code generation in large language models. This framework seeks to improve reliability through adaptive iteration and targeted test generation, breaking down code into smaller segments and generating edge cases from failed tests using runtime feedback. These adaptive agents work to enable precise debugging and incremental refinement, with the ultimate goal of enhancing both the reliability and accuracy of generated code.

---

## Key Features

1. **Adaptive Debugging Logic**  
   AdaptCoder dynamically adjusts its debugging process based on runtime feedback. It refines implementations iteratively, generates targeted test cases, and ensures debugging efforts focus on areas most likely to contain errors.

2. **Edge Case Generation**  
   Failed tests are used to generate new, dynamic test cases targeting edge conditions. This approach enhances robustness by addressing previously unseen scenarios.

3. **Iterative Refinement**  
   Debugging is performed incrementally, ensuring errors are systematically identified and resolved step-by-step.

4. **Scalable Debugging**  
   The framework incorporates techniques like selective sampling and token-efficient prompting to make debugging with large language models (LLMs) practical and scalable.

---

## Adaptive Debugging Logic

### 1. Refining Code for Failed Test Cases
- The adaptiveness is primarily seen in how AdaptCoder reacts to failed test cases by refining the code iteratively.
- Failed tests are randomly selected to refine the implementation.
- This ensures the debugging process focuses on actual failures encountered during execution.

```python
for _ in range(max_iters):
    debug_cur_func_impl = prepare_for_debugging(dataset_type, cur_func_impl, item)
    selected_test = random.choice(failed_tests) if failed_tests else None
    messages = gen.adaptcoder_debug(
        item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], model, "", dataset_type, level
    )
    cur_func_impl, cur_messages = gen.adaptcoder_debug_generate(
        item["prompt"], model, cur_func_impl, messages, selected_test, dataset_type
    )
```


### 2. Generating Edge Cases for Robustness
- When unresolved failures persist, new test cases targeting edge conditions are dynamically generated.
- New test cases are created based on the characteristics of failed tests.
- This makes debugging robust to previously unseen scenarios.

```python
if not is_solved:
    for failed_test in failed_tests:
        item['given_tests'].extend(
            generate_new_tests(failed_test, model, item["prompt"], item["entry_point"])
        )
```
### 3. Iterative Passes and Feedback Integration
- Each debugging iteration evaluates the updated implementation, integrates feedback from test results, and continues refinement until all tests pass or the iteration limit is reached.
- Feedback from test results is continuously integrated to refine the code incrementally.
- Iterative passes ensure a systematic resolution of errors.

```python
is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
test_feedback.append(failed_tests)

if is_passing:
    is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
    if is_solved:
        item["solution"] = cur_func_impl
    break
```

### 4. Adaptive Test Case Generation
- This function generates edge cases dynamically based on observed failures.
- The model creates new test cases tailored to the failed test case.
- This adapts the debugging process to target specific problem areas.

```python
def generate_new_tests(failed_test, model, prompt, entry_point):
    new_tests = []
    response = model.generate(f"Generate edge cases for the test case: {failed_test}")
    for new_case in response:
        if entry_point in new_case and 'assert False' not in new_case:
            new_tests.append(new_case)
    return new_tests
```

### 5. Adaptive Debugging Prompt Generation
- Prompts are dynamically adjusted based on observed failures and the current implementation.
- The debugging prompt incorporates the current implementation and specific failed tests.
- This ensures the model focuses on resolving specific issues instead of generating generic solutions.

```python
messages = gen.adaptcoder_debug(
    item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], model, "", dataset_type, level
)
```

### 6. Debugging Iterations and Selective Sampling
- If execution traces become too lengthy, AdaptCoder selectively samples key blocks to ensure token efficiency and focus on high-priority areas.
- This ensures efficient use of LLM tokens.
- This focuses debugging efforts on critical sections of the code.
```python
messages = gen.adaptcoder_debug(
    item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], model, "", dataset_type, level
)
```


---

## Acknowledgments

We are grateful for the contributions of the following open-source projects, which served as the foundation for our work:

- **[LLMDebugger](https://github.com/FloridSleeves/LLMDebugger)**: Core framework and inspiration for our debugging approach.
- **[Reflexion](https://github.com/noahshinn024/reflexion)**: Adapted code for iterative refinement techniques.
- **[staticfg](https://github.com/coetaur0/staticfg)**: Utilized static flow graph generation methods.

---

## Work in Progress 🚧

This repository is under active development, with regular updates expected. Stay tuned for new features and improvements!

