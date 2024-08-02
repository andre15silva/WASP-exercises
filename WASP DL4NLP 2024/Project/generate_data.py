import subprocess
from typing import List, Optional
from datasets import load_dataset

def compute_diff(
    buggy_code: str, fixed_code: str, context_len: Optional[int] = None
) -> List[str]:
    """
    Computes the diff between the buggy and fixed code.
    """
    context_len = (
        context_len
        if context_len is not None
        else max(len(buggy_code), len(fixed_code))
    )
    with open("/tmp/buggy.java","w") as f: f.write(buggy_code+"\n")
    with open("/tmp/fixed_code.java","w") as f: f.write(fixed_code+"\n")
    # we want to ignore whitespace changes with -w which does not exist in difflib.unified_diff
    # with git diff, we even the name of the changed function in the diff, which helps a lot
    cmd = ["git","diff","--patience",f"-U{context_len}", "-w","/tmp/buggy.java","/tmp/fixed_code.java"]
    return subprocess.run(cmd, capture_output=True).stdout.decode("utf-8")

def user_prompt(fixed_function: str) -> str:
    return f"""
### Fixed Function
```java
{fixed_function}
```
"""

def system_prompt() -> str:
    return """You are an assistant designed to help generate synthetic data for fine-tuning a language model for automatic program repair. You will receive a correctly implemented function as input. Your task is to generate:

1. A buggy version of the provided function.
2. A unit test method that exposes the behavioral difference between the buggy and fixed versions.
3. The stack trace or error message resulting from executing that unit test method with the buggy function.

Please ensure the following when generating the output:

- The buggy function should have a realistic and plausible error that one might encounter in actual programming.
- The unit test MUST be a method and directly target the introduced bug. ONLY the unit test method should be provided, not the entire test suite.
- The stack trace or error message must be realistic and show the exact execution of the unit test.
- Do NOT add any comment in the code about the bug, the bug injection or any other information.

Format the output in the following structure:
### Buggy Function
```java
<Buggy function code>
```

### Unit Test
```java
<Unit test code>
```

### Error Message or Stack Trace
```
<Error message or stack trace>
```
"""

def user_prompt_v2(diff: str) -> str:
    return f"""
### Diff between fixed and buggy functions
```diff
{diff}
```
"""

def system_prompt_v2() -> str:
    return """You are an assistant designed to help generate synthetic data for fine-tuning a language model for automatic program repair. You will receive a bug-fixing diff as input. Your task is to generate:

1. A unit test method that exposes the behavioral difference between the buggy and fixed versions.
2. The stack trace or error message resulting from executing the unit test method.

Please ensure the following when generating the output:

- The unit test MUST be a method and directly target the introduced bug. ONLY the unit test method should be provided, not the entire test suite.
- The stack trace or error message must be realistic and show the exact execution of the unit test.
- Do NOT add any comment in the code about the bug, the bug injection or any other information.

Format the output in the following structure:
### Unit Test
```java
<Unit test code>
```

### Error Message or Stack Trace
```
<Error message or stack trace>
```
"""

import tiktoken

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    enc = tiktoken.encoding_for_model(model)
    n_tokens = len(enc.encode(system_prompt)) + len(enc.encode(user_prompt))
    if n_tokens >= 128000:
        return None

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return completion

def extract_output(completion):
    """
    Extracts the output from the completion object.
    
    Returns:
    - The generated buggy function
    - The generated test case
    - The generated test error or stack trace
    """
    try:
        message = completion.choices[0].message.content
        buggy_function = message.split("### Buggy Function")[1].split("```java")[1].split("```")[0].strip()
        test_case = message.split("### Unit Test")[1].split("```java")[1].split("```")[0].strip()
        error_message = message.split("### Error Message or Stack Trace")[1].split("```")[1].strip()
        return buggy_function, test_case, error_message
    except Exception as e:
        return None, None, None

def extract_output_v2(completion):
    """
    Extracts the output from the completion object.
    
    Returns:
    - The generated test case
    - The generated test error or stack trace
    """
    message = completion.choices[0].message.content

    try:
        test_case = message.split("### Unit Test")[1].split("```java")[1].split("```")[0].strip()
        error_message = message.split("### Error Message or Stack Trace")[1].split("```")[1].strip()
        return test_case, error_message
    except Exception as e:
        return None, None

import concurrent.futures

def main():
    completions = []
    tasks = []

    megadiff_sf = load_dataset("ASSERT-KTH/megadiff-single-function")
    megadiff_sf = megadiff_sf.map(lambda x: {"short_diff": compute_diff(x["buggy_function"], x["fixed_function"], context_len=3)})

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(megadiff_sf["train"][:]["short_diff"])):
            task = executor.submit(call_openai, system_prompt_v2(), user_prompt_v2(megadiff_sf["train"][i]["short_diff"]), model="gpt-4o-mini")
            tasks.append(task)

        completions = [future.result() for future in tasks]

    outputs = [extract_output_v2(completion) for completion in completions]

    megadiff_sf_plus = megadiff_sf["train"].add_column("generated_test_case", [output[0] for output in outputs]).add_column("generated_error_message", [output[1] for output in outputs]).add_column("completion", [completion.to_dict() for completion in completions])
    megadiff_sf_plus.save_to_disk("megadiff_sf_plus")
    
    
if __name__ == "__main__":
    main()
