# KINETIC Glass UI

A real‑time interactive demo of an online stochastic‑gradient‑descent logistic regression model. The app runs a FastAPI server and serves a glass‑morphism web UI that visualises model weights, sigmoid probability, and a keystroke‑based game.

## Features
- Automatic virtual‑environment activation (via `.venv`).
- Real‑time updates of model predictions and learned weights.
- Beautiful UI built with CSS glass‑morphism, animated orbs, and micro‑animations.
- Simple API endpoints:
  - `POST /api/step` – submit a move and get the next model state.
  - `POST /api/reset` – zero the model weights.
- Fully self‑contained – just run `source .venv/bin/activate && python app.py`.

## Getting Started
```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate
# Install dependencies
pip install fastapi uvicorn pydantic
# Run the app
python app.py
```
Open http://localhost:8001 in a browser.

## License
MIT – feel free to fork, modify, and share!
