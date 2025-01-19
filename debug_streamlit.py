import streamlit as st
import sys
import os

def main():
    print("Python version:", sys.version)
    print("Streamlit version:", st.__version__)
    print("Current working directory:", os.getcwd())
    
    st.write("Hello World!")
    st.write("If you can see this, Streamlit is working!")

if __name__ == "__main__":
    main()
