from typing import List, Dict, Any
from llm_sdk import Small_LLM_Model
import numpy as np
from numpy.typing import NDArray


def add_name_mask(
    selected_func_name: str,
    func_names: List[str],
    vocab_dict: Dict[str, int],
    prompt_logits_np: NDArray[np.float64]
) -> NDArray:
    """
    Applies a mask to logits to ensure the next token forms a valid
    function name.

    Args:
        selected_func_name (str): The function name string built so far.
        func_names (List[str]): List of all valid function names.
        vocab_dict (Dict): Dictionary mapping tokens to their IDs.
        prompt_logits_np (NDArray): The original logits from the LLM.

    Returns:
        NDArray: Masked logits where invalid tokens are set to negative
            infinity.
    """

    logits_np_copy = np.full_like(prompt_logits_np, -np.inf)

    for token, id in vocab_dict.items():
        posible_next_token = selected_func_name + token
        for func_name in func_names:
            if func_name.startswith(posible_next_token):
                logits_np_copy[id] = prompt_logits_np[id]
                break
    return logits_np_copy


def add_numeric_mask(
    parameters: str,
    func_params: List[Any],
    vocab_dict: Dict[str, int],
    prompt_logits_np: NDArray[np.float64]
) -> NDArray:
    """
    Applies a mask to logits to restrict output to numeric characters
    and delimiters.

    Args:
        parameters (str): The parameter string built so far.
        func_params (List): List of function parameter definitions.
        vocab_dict (Dict): Dictionary mapping tokens to their IDs.
        prompt_logits_np (NDArray): The original logits from the LLM.

    Returns:
        NDArray: Masked logits allowing only numeric and control tokens.
    """

    valid_tokens = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", "}", "\""
        ]

    if '.' not in parameters:
        valid_tokens.append('.')
    if len(parameters.strip()) == 0:
        valid_tokens.append('-')

    logits_np_copy = np.full_like(prompt_logits_np, -np.inf)
    for token, id in vocab_dict.items():
        if token.strip() and all(c in valid_tokens for c in token.strip()):
            logits_np_copy[id] = prompt_logits_np[id]
    return logits_np_copy


def sign_mask(
    llm: Small_LLM_Model,
    prompt: str,
    vocab_dict: Dict[str, int]
) -> Any:
    """
    Predicts and returns a sign token ('+' or '-') using logit masking.

    Args:
        llm (Small_LLM_Model): The LLM instance.
        prompt (str): The current prompt string.
        vocab_dict (Dict): Dictionary mapping tokens to their IDs.

    Returns:
        str: The predicted sign token.
    """

    prompt_ids = llm.encode(prompt)
    prompt_logits = llm.get_logits_from_input_ids(
            prompt_ids[0].tolist())
    prompt_logits_np = np.array(prompt_logits)
    prompt_logits_np_copy = np.full_like(prompt_logits_np, -np.inf)
    for token, id in vocab_dict.items():
        if token in ['+', '-']:
            prompt_logits_np_copy[id] = prompt_logits_np[id]

    selected_token_id = int(np.argmax(prompt_logits_np_copy))
    selected_token = llm.decode([selected_token_id])
    return selected_token
