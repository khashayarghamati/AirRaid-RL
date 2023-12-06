# Reinforcement Learning Project with Gymnasium

Welcome to the Reinforcement Learning project designed as a sample for University of Hertfordshire students. This project demonstrates the basics of RL models using the gymnasium library.

## Purpose

The purpose of this project is to provide a hands-on introduction to reinforcement learning concepts for students. It showcases the implementation of an RL model using the gymnasium environment, offering a practical insight into how RL algorithms can be applied to solve problems.

## Key Components

### Agent

The **Agent** in this project represents the learning entity that interacts with the environment. It employs a reinforcement learning algorithm to make decisions and learn from the consequences.

### Environment

The **Environment** is the setting in which the agent operates. In this project, we use the gymnasium library to create RL environments, providing a standardized interface for defining tasks.

### Action

**Action** refers to the moves or decisions that the agent can take in the environment. These actions influence the state of the environment and, consequently, the rewards received by the agent.

### Q-function

The **Q-function** (Quality function) is a key concept in reinforcement learning. It estimates the expected future rewards of taking a particular action in a given state. The agent uses the Q-function to make decisions that maximize its cumulative reward over time.

### Policy Function

The **Policy function** defines the strategy that the agent follows to select actions in different states. It can be deterministic or stochastic, specifying the probability distribution over actions given a state.

## Getting Started

Follow these steps to set up and run the project:

### 1. Create a Conda Environment

```bash
# Create a conda environment (replace 'env' with your desired environment name)
conda create --name env python=3.8

# Activate the conda environment
conda activate env
```

### 2. Install Dependencies

Ensure you have Conda installed. Then, create the environment and install the required packages from `environment.yml`.

```bash
conda env create -f environment.yml
```

### 3. Run the Project

Execute the `run.py` file to start the RL model.

```bash
python run.py
```

This command will run the project, and you should see the RL model in action.

## Additional Information

- This project uses the gymnasium library for RL environments.
- Feel free to explore and modify the code to experiment with different RL algorithms.

Happy coding!
