# Flood-It AI: Deep Reinforcement Learning Game

A competitive Flood-It game where you play against an AI trained using Deep Q-Network (DQN) from Stable Baselines3.

## Setup Instructions

### Prerequisites

- Python 3.13+
- Node.js and npm

### Step 1: Install Python Dependencies

```bash
# Install dependencies from pyproject.toml
uv sync
```

### Step 2: Install Frontend Dependencies

```bash
cd frontend
bun run install
```

### Step 3: Train the Model

Before playing, you need to train the AI model:

```bash
cd backend
uv run training/train.py
```

This will:

- Create a DQN model with CNN policy
- Train for 100,000 timesteps (increase to 1M+ for better results)
- Save the model to `backend/models/floodit_dqn.zip`

**Note**: Training may take several minutes. For better AI performance, consider training for 1M+ timesteps by editing `training/train.py`.

### Step 4: Start the Backend API

```bash
cd backend
uv run uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### Step 5: Start the Frontend

In a new terminal:

```bash
cd frontend
bun run dev
```

The React app will be available at `http://localhost:5173` (or the port shown in the terminal)
