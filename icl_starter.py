from vllm import LLM, SamplingParams
from typing import List

def generate_vllm_responses(prompts: List[str], model_name: str = "") -> List[str]:
    """
    Takes a list of prompt strings and returns a list of generated text strings using vLLM.
    """
    # 1. Initialize the LLM (This loads the model weights into GPU memory)
    print(f"Loading model '{model_name}'...")
    llm = LLM(model=model_name)

    # 2. Define sampling parameters
    # You can tweak temperature, max_tokens, top_p, etc. here.
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

    # 3. Run batch generation
    # vLLM automatically handles batching the prompts for maximum GPU throughput
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. Extract the text from the vLLM RequestOutput objects
    # outputs[0] refers to the best completion (if you generated multiple completions per prompt)
    generated_texts = [output.outputs[0].text for output in outputs]

    return generated_texts

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # List of input strings
    my_prompts = [
        "The capital of France is",
        "Write a haiku about a GPU:",
        "Explain the theory of relativity in one sentence:"
    ]
    
    # Get the outputs
    results = generate_vllm_responses(my_prompts)
    
    # Print the results
    for i, (prompt, response) in enumerate(zip(my_prompts, results)):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompt}")
        print(f"Output: {response}")