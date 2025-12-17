# Interactive Privacy-Preserving Chat Demo

This is a web-based interactive demo for the RapLI privacy-preserving LLM framework.

## Prerequisites

- Python 3.8+
- Flask
- Flask-CORS
- PyTorch
- Transformers
- Pandas
- NLTK

## Installation

1. Ensure you have the required packages:
   ```bash
   pip install flask flask-cors torch transformers pandas nltk
   ```

## Running the Demo

1. Navigate to the `interactive_demo` directory:
   ```bash
   cd interactive_demo
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   `http://localhost:5000`

## Features

- **Dataset Selection**: Choose between AG News, SAMSum, Spam Email, etc.
- **Interactive PII Check**: Type a message, and the system will detect PII. You can click on tokens to manually adjust their risk level (0-5).
- **Privacy Sanitization**: The system applies Differential Privacy noise based on the risk levels.
- **LLM Integration**: Simulates sending the sanitized prompt to an LLM and receiving a response.
- **Performance Metrics**: Displays detailed timing for PII detection, risk assessment, noise addition, and total latency.

## Architecture

- `app.py`: Flask backend handling API requests.
- `privacy_engine.py`: Wrapper around the core RapLI modules (`_01pii_detection`, `_02risk_assessment`, `_03dp_noise`).
- `templates/index.html`: Frontend UI using Bootstrap and vanilla JavaScript.
