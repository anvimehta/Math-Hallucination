# Math LLM Hallucination Validator

A web application for validating mathematical claims and detecting hallucinations in Large Language Model (LLM) responses. It uses Wolfram Alpha as the ground truth to check if equations equal claimed values, solve word problems, and compare LLM answers.

---

## What it does

This project provides a chat-based web interface to:

- **Validate Equation Claims**: Check if a mathematical expression equals a target number (e.g., does `sqrt(64) + ln(e^5)` equal 88?). It breaks down the expression step-by-step, identifies where errors occur, and explains discrepancies.

- **Check Word Problems**: Compare LLM responses to Wolfram Alpha's answers for math word problems, integrals, algebra, etc.

- **Ask for Equations**: Have the LLM suggest an equation that equals a target number, then validate it.

- **Built-in Question Bank**: Run a set of predefined math questions and validate answers.

Results are saved in the `wolfram_data/` directory for analysis.

---

## Prerequisites

- Python 3.7+
- A Wolfram Alpha App ID (free at [developer.wolfram.com](https://developer.wolframalpha.com))
- OpenAI API Key (optional, required for LLM-based features like word problem checking or equation suggestions)

---

## Installation and Setup

1. **Clone or download the project** to your local machine.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get API Keys**:
   - Get an OpenAI API key from [platform.openai.com](https://platform.openai.com) if you want to use LLM features:
     1. Go to [platform.openai.com](https://platform.openai.com) and sign up for an account (or log in if you have one).
     2. Navigate to the API section and create a new API key.
     3. Copy the key (keep it secure, as it allows API access).

4. **Configure API Keys**:
   - Create a `.env` file in the project root with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
---

## Running the Application

1. **Start the server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://127.0.0.1:5000`.

3. **Use the web interface**:
   - Enter your queries in the chat interface.
   - Choose modes like validating an equation claim, checking a word problem, or asking for an equation.
   - View results, including step-by-step breakdowns and error analysis.

---

## Usage Examples

- **Equation Claim**: Input a target number and expression, e.g., target 88 for `sqrt(6400) + 2*(sin(pi/2)*24) + ln(e^5)`. Get verdict, step-by-step evaluation, and error location.

- **Word Problem**: Input a math problem, e.g., "What is the integral of x^2 dx?". Compare LLM answer to Wolfram's ground truth.

- **Ask Equation**: Provide a target number, have the LLM suggest an equation, then validate it.

Supported operations in expressions: `+`, `-`, `*`, `/`, `**` (or `^`), `sqrt`, `ln`/`log`, `cos`, `sin`, `tan`, `e`, `pi`.

---

## Data and Logs

- Results are stored in `wolfram_data/` as JSON files (e.g., `equation_claim_log.json`).
- Includes logs for reasoning, validation results, and cached ground truths.

---

## Additional Documentation

- `README_VALIDATOR.md`: Detailed explanation of modes, reasoning logic, and internal workings.
- `wolfram_alpha.py`: Wolfram API integration.
- `math_hallucination_validator.py`: Core validation logic.
- `local_llm_math_checker.py`: LLM interaction for checking answers.

---

## Troubleshooting

- Ensure API keys are correctly set in `.env` and `wolfram_alpha.py`.
- If Wolfram queries fail, check your App IDs and internet connection.
- For LLM features, verify `OPENAI_API_KEY` is set.
