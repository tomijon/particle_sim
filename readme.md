# Simple Particle Simulation

## Overview

A simple particle simulation written in Python. Particles are spawned anywhere on the screen and are given a coordinate to "orbit". Particles move towards the orbit point they have been assigned. It can handle up to approximately 300,000 particles.

This project was practice for writing compute shaders and for writing highly parallelised problems.

## Screenshots

<img src="/Screenshots/1.png" style="zoom:50%;"/><img src="/Screenshots/2.png" style="zoom:50%;"/>

## Technical Stuff

Particle position and velocity are updated using a kernel shader. The particles are moved to the GPU when computing their new positions and then moved back to create the frame in pygame. One optimisation would be to also draw the frame on the GPU to prevent the necessary movement of data.

## Requirements

### List

- pygame
- numpy
- pyopencl

### Installation

```cmd
pip install -r "requirements.txt"
```

 