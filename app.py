from flask import Flask, render_template, request, jsonify, redirect, url_for
import time
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re

from models import (
    call_gpt4, call_gpt4o, call_gpt35_turbo, call_anthropic, call_gemini, 
    MODEL_DISPLAY_NAMES, call_openai_for_token_visualization,
    get_sentence_transformer_embeddings, call_openai_for_prompt_crafting,
    call_anthropic_for_prompt_crafting, call_gemini_for_prompt_crafting
)

from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = Flask(__name__)

# Constants
MAX_CONCURRENT_MODELS = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # Open-source embedding model

# Score interpretation thresholds
SCORE_EXCELLENT_THRESHOLD = 70
SCORE_FAIR_THRESHOLD = 35
PERPLEXITY_EXCELLENT_THRESHOLD = 0.3
PERPLEXITY_GOOD_THRESHOLD = 0.6


# ==================== Helper Functions ====================

def validate_required_fields(data, required_fields):
    """Validate that required fields are present and not empty."""
    for field in required_fields:
        value = data.get(field, '').strip() if isinstance(data.get(field), str) else data.get(field)
        if not value:
            return False, f'{field.replace("_", " ").title()} is required'
    return True, None


def error_response(message, code=400):
    """Create standardized error response."""
    return jsonify({'error': message}), code

# Model function mapping
MODEL_FUNCTIONS = {
    'gpt4': call_gpt4,
    'gpt4o': call_gpt4o,
    'gpt35-turbo': call_gpt35_turbo,
    'anthropic': call_anthropic,
    'gemini': call_gemini
}


# ==================== Metrics Calculation Functions ====================

def calculate_rouge(reference, generated):
    """Calculate ROUGE scores between reference and generated text."""
    if not reference or not generated:
        return None
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        
        return {
            'rouge1': round(scores['rouge1'].fmeasure, 3),
            'rouge2': round(scores['rouge2'].fmeasure, 3),
            'rougeL': round(scores['rougeL'].fmeasure, 3)
        }
    except Exception:
        return None


def get_embeddings(texts):
    """Get embeddings for one or more texts using open-source Sentence Transformers."""
    return get_sentence_transformer_embeddings(texts, EMBEDDING_MODEL)


def calculate_semantic_similarity(reference, generated):
    """
    Calculate semantic similarity using open-source Sentence Transformer embeddings.
    Uses cosine similarity between embedding vectors.
    Returns value between 0-1 where higher = more similar.
    """
    if not reference or not generated:
        return None
    
    embeddings = get_embeddings([reference, generated])
    if not embeddings or len(embeddings) != 2:
        return None
    
    try:
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        return round(float(similarity), 3)
    except Exception as e:
        print(f"Error calculating semantic similarity: {str(e)}")
        return None


def sigmoid(x):
    """Apply standard sigmoid function to normalize values between 0 and 1."""
    return 1 / (1 + math.exp(-x))


def calculate_perplexity(logprobs):
    """
    Calculate perplexity from log probabilities and normalize to 0-1 range using standard sigmoid.
    Lower raw perplexity = better (model is more confident).
    After normalization: closer to 0 = better, closer to 1 = worse.
    """
    if not logprobs or len(logprobs) == 0:
        return None
    
    try:
        avg_neg_logprob = -sum(logprobs) / len(logprobs)
        perplexity = math.exp(avg_neg_logprob)
        
        # Use standard sigmoid on negative perplexity to invert scale
        # (lower perplexity = higher confidence = closer to 0)
        normalized = sigmoid(-perplexity)
        return round(normalized, 2)
    except Exception:
        return None


def count_syllables(word):
    """Count syllables in a word."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    
    # Adjust for silent 'e'
    if word.endswith('e'):
        count -= 1
    
    return max(count, 1)


def calculate_flesch_kincaid(text):
    """Calculate Flesch-Kincaid readability score."""
    if not text:
        return None
    
    try:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return None
        
        words = text.split()
        num_words = len(words)
        
        if num_words == 0:
            return None
        
        num_syllables = sum(count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        score = max(0, min(100, score))
        
        return round(score, 1)
    except Exception:
        return None


def calculate_wobble_score(responses):
    """
    Calculate wobble score (similarity between responses) using embeddings.
    Higher score = more consistent responses (more similar).
    """
    if len(responses) == 0:
        return 0
    if len(responses) == 1:
        return 100
    
    embeddings = get_embeddings(responses)
    if not embeddings:
        # Fallback to simple word-based calculation
        return _calculate_word_based_similarity(responses)
    
    try:
        total_similarity = 0
        comparisons = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                total_similarity += similarity
                comparisons += 1
        
        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0
        return round(avg_similarity * 100)
    except Exception:
        return _calculate_word_based_similarity(responses)


def _calculate_word_based_similarity(responses):
    """Fallback word-based similarity calculation."""
    total_similarity = 0
    comparisons = 0
    
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            words1 = set(responses[i].lower().split())
            words2 = set(responses[j].lower().split())
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union) if len(union) > 0 else 0
            total_similarity += similarity
            comparisons += 1
    
    avg_similarity = total_similarity / comparisons if comparisons > 0 else 0
    return round(avg_similarity * 100)


def calculate_lexical_diversity(responses):
    """
    Calculate lexical diversity (word variety) across responses.
    Higher score = more unique words / less repetition.
    """
    if len(responses) == 0:
        return 0
    if len(responses) == 1:
        words = responses[0].lower().split()
        if len(words) == 0:
            return 0
        unique_words = len(set(words))
        return round(len(words) / unique_words * 10) / 10 if unique_words > 0 else 0
    
    total_diversity = 0
    comparisons = 0
    
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            words1 = set(responses[i].lower().split())
            words2 = set(responses[j].lower().split())
            
            union = words1.union(words2)
            intersection = words1.intersection(words2)
            
            unique_ratio = (len(union) - len(intersection)) / len(union) if len(union) > 0 else 0
            diversity = unique_ratio * len(union)
            
            total_diversity += diversity
            comparisons += 1
    
    avg_diversity = total_diversity / comparisons if comparisons > 0 else 0
    return round(avg_diversity, 2)


def calculate_actual_perplexity_from_runs(results):
    """
    Calculate average perplexity from multiple runs using logprobs, normalized to 0-1 using standard sigmoid.
    Returns None if logprobs not available.
    """
    perplexities = []
    
    for result in results:
        logprobs = result.get('logprobs')
        if logprobs and len(logprobs) > 0:
            try:
                avg_neg_logprob = -sum(logprobs) / len(logprobs)
                perplexity = math.exp(avg_neg_logprob)
                perplexities.append(perplexity)
            except Exception:
                continue
    
    if len(perplexities) == 0:
        return None
    
    avg_perplexity = sum(perplexities) / len(perplexities)
    
    # Use standard sigmoid on negative perplexity to invert scale
    # (lower perplexity = higher confidence = closer to 0)
    normalized = sigmoid(-avg_perplexity)
    return round(normalized, 2)


def get_score_interpretation(score):
    """Get text interpretation and CSS class for a stability score (higher = better)."""
    if score >= SCORE_EXCELLENT_THRESHOLD:
        return 'excellent', 'score-excellent'
    elif score >= SCORE_FAIR_THRESHOLD:
        return 'fair', 'score-fair'
    else:
        return 'poor', 'score-poor'


# ==================== Analysis Functions ====================

def generate_analysis_summary(results):
    """Generate a comprehensive analysis summary from LLM comparison results."""
    successful = [r for r in results if r.get('response') and not r.get('error')]
    
    if not successful:
        return "No successful responses to analyze. Please check your API keys and try again."
    
    models = [r['model'] for r in successful]
    times = [r['time_taken'] for r in successful]
    readability_scores = [r.get('readability') for r in successful if r.get('readability') is not None]
    
    # Find fastest and slowest
    fastest_idx = times.index(min(times))
    fastest_model = models[fastest_idx]
    fastest_time = times[fastest_idx]
    
    slowest_idx = times.index(max(times))
    slowest_model = models[slowest_idx]
    slowest_time = times[slowest_idx]
    
    summary_parts = []
    
    # Speed analysis
    summary_parts.append(f"{fastest_model} was the fastest ({fastest_time}s)")
    if len(successful) > 1:
        summary_parts.append(f"while {slowest_model} took {slowest_time}s")
    
    # Readability analysis
    if readability_scores:
        avg_readability = sum(readability_scores) / len(readability_scores)
        if avg_readability >= 60:
            summary_parts.append(f"The average readability score was {avg_readability:.1f} (easy to read)")
        elif avg_readability >= 30:
            summary_parts.append(f"The average readability score was {avg_readability:.1f} (moderate difficulty)")
        else:
            summary_parts.append(f"The average readability score was {avg_readability:.1f} (complex)")
    
    # Metrics analysis
    has_metrics = any(r.get('rouge') is not None for r in successful)
    
    if has_metrics:
        rouge_scores = [r['rouge']['rouge1'] for r in successful if r.get('rouge')]
        sem_scores = [r.get('semantic_similarity') for r in successful if r.get('semantic_similarity') is not None]
        
        if rouge_scores:
            best_rouge_idx = rouge_scores.index(max(rouge_scores))
            best_rouge_model = models[best_rouge_idx]
            summary_parts.append(f"{best_rouge_model} had the highest ROUGE-1 score ({max(rouge_scores):.3f}), indicating better word overlap with the reference")
        
        if sem_scores:
            best_sem_idx = sem_scores.index(max(sem_scores))
            best_sem_model = models[best_sem_idx]
            avg_similarity = sum(sem_scores) / len(sem_scores)
            summary_parts.append(f"Average semantic similarity was {avg_similarity:.3f}, with {best_sem_model} scoring highest")
    else:
        summary_parts.append("Provide a reference response to see detailed quality metrics like ROUGE scores and semantic similarity")
    
    return ". ".join(summary_parts) + "."


def add_metrics_to_results(results, reference_response):
    """Add all metrics to result objects."""
    for result in results:
        if result['response'] and not result['error']:
            result['readability'] = calculate_flesch_kincaid(result['response'])
            result['perplexity'] = calculate_perplexity(result.get('logprobs'))
            
            if reference_response and reference_response.strip():
                result['rouge'] = calculate_rouge(reference_response, result['response'])
                result['semantic_similarity'] = calculate_semantic_similarity(reference_response, result['response'])
            else:
                result['rouge'] = None
                result['semantic_similarity'] = None
        else:
            result['readability'] = None
            result['rouge'] = None
            result['semantic_similarity'] = None
            result['perplexity'] = None


# ==================== Route Handlers ====================

@app.route('/')
def index():
    return redirect(url_for('prompt_crafter'))


@app.route('/llm-compare')
def llm_compare():
    return render_template('llm_compare.html', active_tab='llm-compare')


@app.route('/prompt-crafter')
def prompt_crafter():
    return render_template('prompt_crafter.html', active_tab='prompt-crafter')


@app.route('/response-stability')
def response_stability():
    return render_template('response_stability.html', active_tab='response-stability')


@app.route('/token-visualizer')
def token_visualizer():
    return render_template('token_visualizer.html', active_tab='token-visualizer')


@app.route('/compare', methods=['POST'])
def compare_models():
    data = request.get_json()
    if not data:
        return error_response('No JSON data provided')
        
    prompt = data.get('prompt', '')
    reference_response = data.get('reference', '')
    selected_models = data.get('models', [])
    parameters = data.get('parameters', {})
    
    if not prompt.strip():
        return error_response('Prompt is required')
    
    if not selected_models:
        return error_response('At least one model must be selected')
    
    # Validate API keys
    for model in selected_models:
        if model in parameters:
            api_key = parameters[model].get('api_key', '')
            if not api_key.strip():
                return error_response(f'API key is required for {model}')
    
    # Execute LLM calls concurrently
    results = []
    max_workers = min(MAX_CONCURRENT_MODELS, len(selected_models))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {}
        for model in selected_models[:MAX_CONCURRENT_MODELS]:
            if model in MODEL_FUNCTIONS and model in parameters:
                future = executor.submit(
                    MODEL_FUNCTIONS[model],
                    prompt,
                    parameters[model]['api_key'],
                    parameters[model]
                )
                future_to_model[future] = model
        
        for future in as_completed(future_to_model):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                model = future_to_model[future]
                results.append({
                    'model': MODEL_DISPLAY_NAMES.get(model, model),
                    'response': None,
                    'time_taken': 0,
                    'error': f"Execution error: {str(e)}"
                })
    
    # Add metrics to results
    add_metrics_to_results(results, reference_response)
    
    # Sort results by model name
    results.sort(key=lambda x: x['model'])
    
    # Generate analysis summary
    analysis_summary = generate_analysis_summary(results)
    
    return jsonify({
        'results': results,
        'analysis_summary': analysis_summary
    })


@app.route('/check-stability', methods=['POST'])
def check_stability():
    """Run the same prompt multiple times to check response stability."""
    data = request.json
    
    # Validate required fields
    is_valid, error_msg = validate_required_fields(data, ['prompt', 'model', 'api_key'])
    if not is_valid:
        return error_response(error_msg)
    
    prompt = data.get('prompt', '').strip()
    model = data.get('model', '')
    api_key = data.get('api_key', '').strip()
    num_runs = data.get('num_runs', 20)
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 500)
    
    # Validate num_runs
    try:
        num_runs = int(num_runs)
        if num_runs < 5 or num_runs > 30:
            return error_response('Number of runs must be between 5 and 30')
    except ValueError:
        return error_response('Invalid number of runs')
    
    if model not in MODEL_FUNCTIONS:
        return error_response(f'Invalid model: {model}')
    
    # Run the model multiple times
    responses = []
    response_texts = []
    parameters = {
        'temperature': temperature,
        'max_tokens': max_tokens
    }
    
    for i in range(num_runs):
        try:
            result = MODEL_FUNCTIONS[model](prompt, api_key, parameters)
            
            individual_perplexity = None
            if result.get('logprobs'):
                individual_perplexity = calculate_perplexity(result.get('logprobs'))
            
            responses.append({
                'run': i + 1,
                'response': result.get('response', ''),
                'time_taken': result.get('time_taken', 0),
                'logprobs': result.get('logprobs'),
                'tokens': result.get('tokens'),
                'perplexity': individual_perplexity,
                'error': result.get('error')
            })
            
            if result.get('response') and not result.get('error'):
                response_texts.append(result['response'])
                
        except Exception as e:
            responses.append({
                'run': i + 1,
                'response': None,
                'time_taken': 0,
                'logprobs': None,
                'error': f"Error: {str(e)}"
            })
    
    # Calculate metrics
    if len(response_texts) > 0:
        wobble_score = calculate_wobble_score(response_texts)
        lexical_diversity = calculate_lexical_diversity(response_texts)
        actual_perplexity = calculate_actual_perplexity_from_runs(responses)
    else:
        wobble_score = 0
        lexical_diversity = 0
        actual_perplexity = None
    
    # Get interpretations
    wobble_text, wobble_class = get_score_interpretation(wobble_score)
    
    # For lexical diversity, lower is better (more consistent)
    if lexical_diversity >= 70:
        diversity_text, diversity_class = 'poor', 'score-poor'
    elif lexical_diversity >= 35:
        diversity_text, diversity_class = 'fair', 'score-fair'
    else:
        diversity_text, diversity_class = 'excellent', 'score-excellent'
    
    # Perplexity interpretation (normalized 0-1, lower is better)
    if actual_perplexity is not None:
        if actual_perplexity <= PERPLEXITY_EXCELLENT_THRESHOLD:
            perplexity_text, perplexity_class = 'excellent', 'score-excellent'
        elif actual_perplexity <= PERPLEXITY_GOOD_THRESHOLD:
            perplexity_text, perplexity_class = 'good', 'score-fair'
        else:
            perplexity_text, perplexity_class = 'fair', 'score-poor'
    else:
        perplexity_text, perplexity_class = 'N/A', 'score-fair'
    
    # Generate summary
    model_name = MODEL_DISPLAY_NAMES.get(model, model)
    successful_runs = len(response_texts)
    summary_text = f"Completed {successful_runs} of {num_runs} runs for {model_name}"
    
    return jsonify({
        'responses': responses,
        'total_runs': num_runs,
        'model': model,
        'wobble_score': wobble_score,
        'wobble_class': wobble_class,
        'wobble_text': wobble_text,
        'lexical_diversity': lexical_diversity,
        'diversity_class': diversity_class,
        'diversity_text': diversity_text,
        'actual_perplexity': actual_perplexity,
        'perplexity_class': perplexity_class,
        'perplexity_text': perplexity_text,
        'summary_text': summary_text
    })


@app.route('/visualize-tokens', methods=['POST'])
def visualize_tokens():
    """Generate token visualization data with top alternatives."""
    data = request.json
    
    # Validate required fields
    is_valid, error_msg = validate_required_fields(data, ['prompt', 'model', 'api_key'])
    if not is_valid:
        return error_response(error_msg)
    
    prompt = data.get('prompt', '').strip()
    model = data.get('model', '')
    api_key = data.get('api_key', '').strip()
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 500)
    
    # Only support OpenAI models for token visualization
    model_mapping = {
        'gpt4': ('gpt-4', 'gpt4'),
        'gpt4o': ('gpt-4o', 'gpt4o'),
        'gpt35-turbo': ('gpt-3.5-turbo', 'gpt35-turbo')
    }
    
    if model not in model_mapping:
        return error_response(f'Model {model} does not support token visualization. Please select an OpenAI model.')
    
    model_name, model_key = model_mapping[model]
    
    # Build parameters dictionary with basic and advanced options
    parameters = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_logprobs': data.get('top_logprobs', 5)
    }
    
    # Add advanced parameters if provided (they are optional)
    if 'top_p' in data and data['top_p'] is not None:
        parameters['top_p'] = data['top_p']
    
    if 'frequency_penalty' in data and data['frequency_penalty'] is not None:
        parameters['frequency_penalty'] = data['frequency_penalty']
    
    if 'presence_penalty' in data and data['presence_penalty'] is not None:
        parameters['presence_penalty'] = data['presence_penalty']
    
    try:
        result = call_openai_for_token_visualization(model_name, model_key, prompt, api_key, parameters)
        return jsonify(result)
    except Exception as e:
        return error_response(f'Token visualization failed: {str(e)}', 500)


@app.route('/craft-prompt-mock', methods=['POST'])
def craft_prompt_real():
    """Generate prompt crafting using user-selected model."""
    data = request.json
    mode = data.get('mode', 'build')
    
    # Validate required fields
    is_valid, error_msg = validate_required_fields(data, ['model', 'api_key'])
    if not is_valid:
        return error_response(error_msg)
    
    model = data.get('model')
    api_key = data.get('api_key', '').strip()
    
    if not api_key:
        return error_response('API key is required')
    
    try:
        # Build the instruction based on mode
        if mode == 'build':
            task_description = data.get('task_description', '').strip()
            purpose = data.get('purpose', '').strip()
            output_format = data.get('output_format', '').strip()
            style_tone = data.get('style_tone', '').strip()
            
            if not task_description:
                return error_response('Task description is required')
            
            instruction = f"""You are a prompt engineering expert. Create an optimized prompt based on the following components:

Task Description: {task_description}
{f"Purpose: {purpose}" if purpose else ""}
{f"Output Format: {output_format}" if output_format else ""}
{f"Style/Tone: {style_tone}" if style_tone else ""}

Your task:
1. Craft a clear, effective, and detailed prompt that incorporates the provided components
2. Make it specific, actionable, and well-structured
3. The prompt must only contain information from the information provided above. Do not make up new information
Return your response in this exact format:

CRAFTED PROMPT:
[Your crafted prompt here - make it comprehensive and ready to use. Follow best principles, with markdown syntax for different parts of the prompt]

SUGGESTIONS:
- [List 3-5 suggestions for further improvement, one per line]
- [Each suggestion should be actionable and specific, including whether the new information required is a task description, purpose, output format, or style/tone]
- [Focus on what could be added to make the prompt even better]"""

        else:  # feedback mode
            existing_prompt = data.get('existing_prompt', '').strip()
            
            if not existing_prompt:
                return error_response('Existing prompt is required')
            
            instruction = f"""You are a prompt engineering expert. Analyze and improve the following prompt:

ORIGINAL PROMPT:
{existing_prompt}

Your task:
1. Identify what's good about this prompt
2. Identify what's missing or could be improved
3. Create an optimized version that is more effective, clear, and detailed
4. Add necessary structure, format instructions, tone specifications, and context

Return your response in this exact format:

OPTIMIZED PROMPT:
[Your optimized prompt here - make it comprehensive and ready to use. Follow best principles, with markdown syntax for different parts of the prompt. Make sure markdown format is used to highlight areas of information]

ENHANCEMENTS:
- [List each enhancement you made, one per line]
- [Be specific about what you added or changed]
- [Focus on improvements like clarity, structure, specificity, format, tone, etc.]"""

        # Call the appropriate API based on model selection using centralized functions
        result_text = None
        time_taken = 0
        
        if model in ['gpt4', 'gpt4o', 'gpt35-turbo']:
            # OpenAI models
            model_name_map = {
                'gpt4': 'gpt-4',
                'gpt4o': 'gpt-4o',
                'gpt35-turbo': 'gpt-3.5-turbo'
            }
            result_text, time_taken = call_openai_for_prompt_crafting(
                model_name_map[model],
                instruction,
                api_key
            )
            
        elif model == 'anthropic':
            # Anthropic Claude
            result_text, time_taken = call_anthropic_for_prompt_crafting(
                instruction,
                api_key
            )
                
        elif model == 'gemini':
            # Google Gemini
            result_text, time_taken = call_gemini_for_prompt_crafting(
                instruction,
                api_key
            )
            
        else:
            return error_response(f'Invalid model: {model}')
        
        if not result_text:
            return error_response('No response generated from the model')
        
        # Parse response
        if mode == 'build':
            split_marker = "SUGGESTIONS:"
            title = 'Crafted Prompt'
            suggestions_title = 'Ways to Improve This Prompt'
        else:
            split_marker = "ENHANCEMENTS:"
            title = 'Optimized Prompt'
            suggestions_title = 'Enhancements'
        
        crafted_prompt = ""
        suggestions = []
        
        if "PROMPT:" in result_text and split_marker in result_text:
            parts = result_text.split(split_marker)
            crafted_part = parts[0].split("PROMPT:")[1].strip()
            suggestions_part = parts[1].strip()
            
            crafted_prompt = crafted_part
            
            # Parse suggestions
            for line in suggestions_part.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    suggestions.append(line[1:].strip())
        else:
            crafted_prompt = result_text
            suggestions = ["The AI provided suggestions in a different format."]
        
        return jsonify({
            'crafted_prompt': crafted_prompt,
            'suggestions': suggestions,
            'title': title,
            'suggestions_title': suggestions_title,
            'time_taken': time_taken
        })
            
    except Exception as e:
        return error_response(f'Prompt crafting failed: {str(e)}', 500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
