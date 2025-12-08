import time
import math
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# Constants
EULER = math.e

# Model display names mapping
MODEL_DISPLAY_NAMES = {
    'gpt4': 'OpenAI GPT-4',
    'gpt4o': 'OpenAI GPT-4o',
    'gpt35-turbo': 'OpenAI GPT-3.5-turbo',
    'anthropic': 'Anthropic Claude',
    'gemini': 'Google Gemini Flash 2.0'
}

# Initialize embedding model at startup
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

def _initialize_embedding_model():
    """Initialize the sentence transformer model. Called once at startup."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {_EMBEDDING_MODEL_NAME}...")
            _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load embedding model: {str(e)}")
            _EMBEDDING_MODEL = None

# Load the model when the module is imported
_initialize_embedding_model()


def logprob_to_percentage(logprob):
    """Convert log probability to percentage."""
    return round(100 * (EULER ** logprob), 2)


def _extract_openai_logprobs(response):
    """Extract logprobs and tokens from OpenAI response."""
    logprobs_data = None
    tokens_data = None
    
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        logprobs_data = [
            token_data.logprob 
            for token_data in response.choices[0].logprobs.content
        ]
        tokens_data = [
            token_data.token
            for token_data in response.choices[0].logprobs.content
        ]
    
    return logprobs_data, tokens_data


def _extract_detailed_token_data(response):
    """Extract detailed token information including top alternatives for visualization."""
    token_data = []
    
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        for idx, token_info in enumerate(response.choices[0].logprobs.content):
            # Get the chosen token
            chosen_token = token_info.token
            chosen_logprob = token_info.logprob
            
            # Get top alternatives
            alternatives = []
            if token_info.top_logprobs:
                for alt in token_info.top_logprobs:
                    if alt.token != chosen_token:  # Don't include the chosen token in alternatives
                        alternatives.append({
                            'token': alt.token,
                            'logprob': alt.logprob,
                            'probability': logprob_to_percentage(alt.logprob)
                        })
            
            token_data.append({
                'position': idx,
                'chosen_token': chosen_token,
                'chosen_logprob': chosen_logprob,
                'chosen_probability': logprob_to_percentage(chosen_logprob),
                'alternatives': alternatives
            })
    
    return token_data


def _create_response_dict(model_key, response_text, time_taken, logprobs=None, tokens=None, error=None):
    """Create standardized response dictionary."""
    return {
        'model': MODEL_DISPLAY_NAMES.get(model_key, model_key),
        'response': response_text,
        'time_taken': round(time_taken, 2),
        'logprobs': logprobs,
        'tokens': tokens,
        'error': error
    }


def _call_openai_model(model_name, model_key, prompt, api_key, parameters):
    """Generic OpenAI API caller for all OpenAI models."""
    start_time = time.time()
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=parameters.get('temperature', 0.7),
            max_tokens=parameters.get('max_tokens', 500),
            logprobs=True,
            top_logprobs=1
        )
        
        end_time = time.time()
        logprobs_data, tokens_data = _extract_openai_logprobs(response)
        
        return _create_response_dict(
            model_key,
            response.choices[0].message.content,
            end_time - start_time,
            logprobs_data,
            tokens_data
        )
    except Exception as e:
        end_time = time.time()
        return _create_response_dict(
            model_key,
            None,
            end_time - start_time,
            error=f"Error: {str(e)}"
        )


def call_gpt4(prompt, api_key, parameters):
    """Call OpenAI GPT-4 API with timing and error handling."""
    return _call_openai_model("gpt-4", "gpt4", prompt, api_key, parameters)


def call_gpt4o(prompt, api_key, parameters):
    """Call OpenAI GPT-4o API with timing and error handling."""
    return _call_openai_model("gpt-4o", "gpt4o", prompt, api_key, parameters)


def call_gpt35_turbo(prompt, api_key, parameters):
    """Call OpenAI GPT-3.5-turbo API with timing and error handling."""
    return _call_openai_model("gpt-3.5-turbo", "gpt35-turbo", prompt, api_key, parameters)


def call_anthropic(prompt, api_key, parameters):
    """Call Anthropic Claude API with timing and error handling."""
    start_time = time.time()
    try:
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=parameters.get('max_tokens', 500),
            temperature=parameters.get('temperature', 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        
        end_time = time.time()
        
        # Extract text from response
        response_text = ""
        if response.content:
            text_parts = [
                content_block.text 
                for content_block in response.content 
                if content_block.type == 'text'
            ]
            response_text = ''.join(text_parts)
        
        return _create_response_dict(
            'anthropic',
            response_text or "No response generated",
            end_time - start_time
        )
    except Exception as e:
        end_time = time.time()
        return _create_response_dict(
            'anthropic',
            None,
            end_time - start_time,
            error=f"Error: {str(e)}"
        )


def call_gemini(prompt, api_key, parameters):
    """Call Google Gemini Flash 2.0 API with timing and error handling."""
    start_time = time.time()
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        generation_config = {
            'temperature': parameters.get('temperature', 0.7),
            'max_output_tokens': parameters.get('max_tokens', 500)
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        end_time = time.time()
        
        return _create_response_dict(
            'gemini',
            response.text or "No response generated",
            end_time - start_time
        )
    except Exception as e:
        end_time = time.time()
        return _create_response_dict(
            'gemini',
            None,
            end_time - start_time,
            error=f"Error: {str(e)}"
        )


def call_openai_for_token_visualization(model_name, model_key, prompt, api_key, parameters):
    """
    Call OpenAI API specifically for token visualization with detailed logprobs.
    Returns token-by-token breakdown with configurable number of alternatives.
    """
    start_time = time.time()
    try:
        client = OpenAI(api_key=api_key)
        
        # Build API parameters
        api_params = {
            'model': model_name,
            'messages': [{"role": "user", "content": prompt}],
            'temperature': parameters.get('temperature', 0.7),
            'max_tokens': parameters.get('max_tokens', 500),
            'logprobs': True,
            'top_logprobs': parameters.get('top_logprobs', 5)
        }
        
        # Add optional advanced parameters if provided
        if 'top_p' in parameters and parameters['top_p'] is not None:
            api_params['top_p'] = parameters['top_p']
        
        if 'frequency_penalty' in parameters and parameters['frequency_penalty'] is not None:
            api_params['frequency_penalty'] = parameters['frequency_penalty']
        
        if 'presence_penalty' in parameters and parameters['presence_penalty'] is not None:
            api_params['presence_penalty'] = parameters['presence_penalty']
        
        response = client.chat.completions.create(**api_params)
        
        end_time = time.time()
        
        # Extract detailed token data
        token_data = _extract_detailed_token_data(response)
        
        return {
            'model': MODEL_DISPLAY_NAMES.get(model_key, model_key),
            'response': response.choices[0].message.content,
            'time_taken': round(end_time - start_time, 2),
            'token_data': token_data,
            'total_tokens': len(token_data),
            'error': None
        }
    except Exception as e:
        end_time = time.time()
        return {
            'model': MODEL_DISPLAY_NAMES.get(model_key, model_key),
            'response': None,
            'time_taken': round(end_time - start_time, 2),
            'token_data': [],
            'total_tokens': 0,
            'error': f"Error: {str(e)}"
        }


def get_sentence_transformer_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L12-v2"):
    """
    Get embeddings for one or more texts using pre-loaded Sentence-Transformers model.
    
    Args:
        texts: String or list of strings to get embeddings for
        model_name: Model name (default: all-MiniLM-L12-v2) - only used if global model not loaded
    
    Returns:
        List of numpy arrays containing embeddings, or None on error
    """
    global _EMBEDDING_MODEL
    
    try:
        import numpy as np
        
        # Use the pre-loaded global model if available, otherwise load it
        model = _EMBEDDING_MODEL
        if model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Warning: Loading model on-demand (should have been loaded at startup)")
            model = SentenceTransformer(model_name)
        
        # Ensure texts is a list
        texts_list = texts if isinstance(texts, list) else [texts]
        
        # Get embeddings
        embeddings = model.encode(texts_list, convert_to_numpy=True)
        
        # Return as list of numpy arrays (consistent with OpenAI format)
        return [np.array(emb) for emb in embeddings]
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return None


def call_openai_for_prompt_crafting(model_name, instruction, api_key, system_prompt=None):
    """
    Call OpenAI API for prompt crafting with custom system prompt.
    
    Args:
        model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo')
        instruction: The user instruction/prompt
        api_key: OpenAI API key
        system_prompt: Optional system prompt (default: prompt engineering expert)
    
    Returns:
        Tuple of (response_text, time_taken) or raises Exception
    """
    start_time = time.time()
    client = OpenAI(api_key=api_key)
    
    if system_prompt is None:
        system_prompt = "You are a prompt engineering expert who helps users create effective prompts for AI systems."
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        temperature=0,
        top_p=0.1,
        max_tokens=800
    )
    
    time_taken = round(time.time() - start_time, 2)
    return response.choices[0].message.content, time_taken


def call_anthropic_for_prompt_crafting(instruction, api_key, system_prompt=None):
    """
    Call Anthropic Claude API for prompt crafting with custom system prompt.
    
    Args:
        instruction: The user instruction/prompt
        api_key: Anthropic API key
        system_prompt: Optional system prompt (default: prompt engineering expert)
    
    Returns:
        Tuple of (response_text, time_taken) or raises Exception
    """
    start_time = time.time()
    client = Anthropic(api_key=api_key)
    
    if system_prompt is None:
        system_prompt = "You are a prompt engineering expert who helps users create effective prompts for AI systems."
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": instruction}]
    )
    
    time_taken = round(time.time() - start_time, 2)
    
    # Extract text from response
    result_text = ""
    if response.content:
        text_parts = [
            content_block.text 
            for content_block in response.content 
            if content_block.type == 'text'
        ]
        result_text = ''.join(text_parts)
    
    return result_text, time_taken


def call_gemini_for_prompt_crafting(instruction, api_key, system_prompt=None):
    """
    Call Google Gemini API for prompt crafting with custom system prompt.
    
    Args:
        instruction: The user instruction/prompt
        api_key: Google API key
        system_prompt: Optional system prompt (default: prompt engineering expert)
    
    Returns:
        Tuple of (response_text, time_taken) or raises Exception
    """
    start_time = time.time()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    if system_prompt is None:
        system_prompt = "You are a prompt engineering expert who helps users create effective prompts for AI systems."
    
    generation_config = {
        'temperature': 0,
        'max_output_tokens': 800
    }
    
    # For Gemini, prepend system prompt to the instruction
    full_prompt = f"{system_prompt}\n\n{instruction}"
    
    response = model.generate_content(
        full_prompt,
        generation_config=generation_config
    )
    
    time_taken = round(time.time() - start_time, 2)
    return response.text, time_taken
