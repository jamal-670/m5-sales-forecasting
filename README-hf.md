# M5 Forecasting Demo

This repository is prepared for deployment on Hugging Face Spaces using Docker.

## Files
- app/app.py: Streamlit entry point
- app.py: root-level entry point for Hugging Face Spaces
- Dockerfile.hf: Docker image for the Space
- requirements-hf.txt: dependencies

## Deploy on Hugging Face Spaces
1. Create a new Space.
2. Choose Docker as the SDK.
3. Upload this repository or connect it to GitHub.
4. Set the container port to 7860.
5. Use Dockerfile.hf as the Dockerfile.
6. Ensure the repository includes both app.py and app/app.py.
