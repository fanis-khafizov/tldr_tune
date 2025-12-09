"""Shared prompt templates for Llama 3.1 Instruct."""

SYSTEM_PROMPT = "Summarize the following Reddit post in one short sentence (TL;DR)."


def format_prompt(post: str) -> str:
    """Format prompt for Llama 3.1 Instruct."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{post.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def format_output(summary: str) -> str:
    """Format output with end token."""
    return f"{summary.strip()}<|eot_id|>"
