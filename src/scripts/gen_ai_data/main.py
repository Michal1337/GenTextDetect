import subprocess

from gen_params import LLMS

SCRIPT_PATH = "natural_questions.py"


def execute_script(script_path: str, llm_name: str, llm_path: str, quant: str) -> None:
    """Execute the script to generate AI-rewritten blogs."""
    cmd = ["python", script_path, llm_name, llm_path]
    if quant:
        cmd.append(quant)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    for llm_name, llm_path, quant in LLMS:
        execute_script(SCRIPT_PATH, llm_name, llm_path, quant=quant)
