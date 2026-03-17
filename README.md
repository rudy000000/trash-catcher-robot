# Trash Catcher Robot

A smart trash can robot that autonomously moves to catch thrown trash before it lands.

## Overview
- **Trajectory Prediction**: Extended Kalman Filter + parabolic model (Drake)
- **Motion Planning**: GCS Trajopt + MPC (Drake)
- **Sim-to-Real**: Gazebo + PPO reinforcement learning (Stable-Baselines3)
- **Hardware**: Omni-wheel base + Raspberry Pi 5 + Stereo Camera

## Motivation
Designed as an MIT portfolio project to demonstrate robotics, control theory, and machine learning integration.

## Tech Stack
- [Drake](https://drake.mit.edu/) — trajectory optimization & control
- ROS2 Humble
- Gazebo
- Stable-Baselines3 (PPO)
- Python 3.12

## Progress
- [x] Phase 1: Environment setup
- [x] Phase 2: Trajectory prediction module (5 unit tests passing)
- [x] Phase 3: Motion planning module (5 unit tests passing)
- [x] Phase 4: Sim-to-Real training (success rate: 20% @ 50k steps)
- [ ] Phase 5: Real robot deployment

## Results
_Coming soon_