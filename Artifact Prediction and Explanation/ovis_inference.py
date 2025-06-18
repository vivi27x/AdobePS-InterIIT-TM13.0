import torch
from PIL import Image
from transformers import AutoModelForCausalLM

def load_ovis():
    """
    Loads the Ovis1.6-Gemma2-9B model and initializes the vision and text tokenizers

    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The loaded Ovis1.6-Gemma2-9B model configured with specified parameters.
            - text_tokenizer (PreTrainedTokenizer): The tokenizer for processing text input associated with the model.
            - visual_tokenizer (PreTrainedTokenizer): The tokenizer for processing visual input associated with the model.
    """

    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                                torch_dtype=torch.bfloat16,
                                                multimodal_max_length=8192,
                                                trust_remote_code=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer


def run_ovis_inference(image, text, model, text_tokenizer, visual_tokenizer):
    """
    Processes an image and a text query using the Ovis1.6-Gemma2-9B model to generate a response.

    Args:
        image (PIL.Image): The input image to be processed by the model.
        text (str): The text query to be combined with the image input for processing.
        model (AutoModelForCausalLM): The preloaded Ovis1.6-Gemma2-9B model.
        text_tokenizer (PreTrainedTokenizer): The tokenizer for processing the text input.
        visual_tokenizer (PreTrainedTokenizer): The tokenizer for processing the visual (image) input.

    Returns:
        str: The generated output text from the model based on the input image and text.
    """
    query = f'<image>\n{text}'

    # Preparing the query and setting up the conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # Generating the output on the given query
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output