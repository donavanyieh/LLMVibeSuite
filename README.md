# LLMVibeSuite
I created this app as a side project as an idea to help people gain a little more intuition of how GenAI output works behind the scenes. As a data scientist, I regularly give workshops/ explain the inconsistancy and non-deterministic outputs of AI. Wanted to make a tool to help visualize some of these outputs live to make things a little more interactive and digestable

# What does this do?
Bunch of tools that are either utility, or interesting to output metadata of different models. We have:

### Prompt Crafter
Simple and straightforward, fill up a few fields and get an optimized prompt. Alternatively, key in existing prompt and get a refined version. Offers area of improvement or asks for more information if necessary

### LLM Comparison
Enter a prompt, and optionally provide a reference ground truth response. Select bunch of models to generate output.
Look at model confidence, runtime, similiarity to reference response (if provided), and output side by side

### Response Stability
Kind of like a voting mechanism. Takes a prompt and runs N times. Look at how results vary, average confidence, and get a stability score. High stability means LLM is confident. Low means LLM might be 'guessing' as different outputs are almost equally likely.

### Token Visualizer
Enter a prompt and look at the output. Break each token down to its logprob. See how top-p and temperature affects the token probabilities, and how sometimes the highest probability token may not get sampled - thanks to sampling behind the scenes. This also shows where the model becomes 'unsure' in a long output<br><br>

# Project Setup
Use python 3.13, create virtual environment, and install uv
```bash
pip install uv
```
install requirements
```bash
uv pip install -r requirements.txt
```
just run the app
```bash
python app.py
```
# Dependencies
As mentioned above, python 3.13<br>
For embedding models, we use sentence-transformers/all-MiniLM-L12-v2 as an open source embedding model. Model is downloaded on initialization.

# Disclaimer
This entire tool is <b>Vibe Coded</b> - almost entirely while I was on a threadmill walking. Shoutout to [Replit](https://replit.com/) for the easy-to-use app, capable coding agents, and easy porting to github, where I ultimately did the QCs.

This also means I will not be maintaining this application - which could potentially break if model endpoints change or gets deprecated.

<b>This application requires API keys to be provided through the UI.</b> I do not collect any data - everything is run locally, except when passing to the model provider itself. <b>Make sure you set billing limits if you are unfamiliar.</b>
