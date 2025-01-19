import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

class LLMPredictor:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.nvidia_url = "http://localhost:8000/predictions"  # Update with your Nvidia Triton server URL
        
    def predict_ollama(self, ticker, prediction_days, model_type, historical_data):
        """Get stock price prediction from Ollama model"""
        try:
            # Format historical data
            recent_prices = historical_data.tail(30).values.tolist()
            current_price = historical_data.iloc[-1]
            
            # Create prompt
            prompt = f"""You are a stock market expert. Given the following data for {ticker}:
            - Current price: ${current_price:.2f}
            - Recent closing prices (last 30 days): {recent_prices}
            
            Please analyze this data and predict the stock price in {prediction_days} days.
            Consider market trends, price patterns, and potential market factors.
            
            Provide your prediction in this exact format:
            PREDICTION: $XX.XX
            REASONING: Your detailed explanation here
            """
            
            # Make API request
            response = requests.post(
                self.ollama_url,
                json={
                    "model": model_type,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                return None, f"Error from Ollama API: {response.text}"
            
            # Parse response
            response_text = response.json().get('response', '')
            
            # Extract prediction and reasoning
            try:
                prediction_line = [line for line in response_text.split('\n') if line.startswith('PREDICTION:')][0]
                reasoning_line = [line for line in response_text.split('\n') if line.startswith('REASONING:')][0]
                
                predicted_price = float(prediction_line.split('$')[1].strip())
                reasoning = reasoning_line.replace('REASONING:', '').strip()
                
                return predicted_price, reasoning
            except Exception as e:
                return None, f"Error parsing Ollama response: {str(e)}"
                
        except Exception as e:
            return None, f"Error in Ollama prediction: {str(e)}"
    
    def predict_nvidia(self, ticker, prediction_days):
        """Get stock price prediction from Nvidia Triton model"""
        try:
            # This is a placeholder for Nvidia Triton integration
            # You would need to implement the actual API call based on your model
            return None, "Nvidia Triton prediction not implemented yet"
        except Exception as e:
            return None, f"Error in Nvidia prediction: {str(e)}"

def get_available_models():
    """Get list of available models"""
    try:
        # Get Ollama models
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []
