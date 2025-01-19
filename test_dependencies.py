import sys
print("Python version:", sys.version)

print("\nTesting streamlit...")
import streamlit as st
print("Streamlit version:", st.__version__)

print("\nTesting yfinance...")
import yfinance as yf
print("yfinance version:", yf.__version__)

print("\nTesting pandas...")
import pandas as pd
print("pandas version:", pd.__version__)

print("\nTesting numpy...")
import numpy as np
print("numpy version:", np.__version__)

print("\nTesting plotly...")
import plotly
print("plotly version:", plotly.__version__)

print("\nTesting scikit-learn...")
import sklearn
print("scikit-learn version:", sklearn.__version__)

print("\nTesting nltk...")
import nltk
print("nltk version:", nltk.__version__)

print("\nTesting textblob...")
import textblob
print("textblob version:", textblob.__version__)

print("\nTesting transformers...")
import transformers
print("transformers version:", transformers.__version__)

print("\nTesting torch...")
import torch
print("torch version:", torch.__version__)

print("\nAll dependencies imported successfully!")
