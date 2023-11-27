import numpy as np
import pygame
from pyopencl import *
from random import randint
from os import environ

# Easy access mem flags.
WRITE_ONLY = mem_flags.WRITE_ONLY
READ_ONLY = mem_flags.READ_ONLY
COPY_HOST_PTR = mem_flags.COPY_HOST_PTR

# Set default mode for pyopencl to use.
environ["PYOPENCL_CTX"] = "0"

# Opencl context.
context = create_some_context()
context_queue = CommandQueue(context)

# Create window.
pygame.init()
window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
width = window.get_width()
height = window.get_height()

# Datatypes
Vector2D = np.dtype([
    ("x", np.int32),
    ("y", np.int32)])

Particle = np.dtype([
    ("position", Vector2D),
    ("velocity", Vector2D),
    ("orbit", Vector2D)])

# GPU Compute Particle function.
##__update_particles = Program(context, """
##typedef struct vector2d {
##    int x;
##    int y;
##} Vector2D;
##
##
##typedef struct particle {
##    Vector2D position;
##    Vector2D velocity;
##    Vector2D orbit;
##} Particle;
##
##
##__kernel void update_particles(
##    __global const Particle *particles,
##    __global const Particle *new_particles)
##{
##    int gid = get_global_id(0);
##
##    
##
##
##
##
##"""


def update_particles(particles) -> np.array:
    """Updates the particles, pulling them towards their center."""
    new_particles = np.zeros(particles.shape, dtype=Particle)

    # GPU Buffers.
    particles_gpu = Buffer(context, READ_ONLY | COPY_HOST_PTR, hostbuf=particles)
    new_particles_gpu = Buffer(context, WRITE_ONLY, new_particles.nbytes)

    __update_particles(context_queue, new_particles.shape, None,
                       particles_gpu, new_particles_gpu)
    enqueue_copy(context_queue, new_particles, new_particles_gpu)
    return new_particles


# Simulation variables.
center = [width//2, height//2]
n_particles = 1000

# Create particles.
particles = np.zeros((n_particles,), dtype=Particle)
for i in range(n_particles):
    particles[i]["position"]["x"] = randint(0, width)
    particles[i]["position"]["y"] = randint(0, height)
    particles[i]["velocity"]["x"] = randint(0, 5)
    particles[i]["velocity"]["y"] = randint(0, 5)
    particles[i]["orbit"]["x"] = width // 2
    particles[i]["orbit"]["y"] = height // 2

    print(particles[i])


pygame.quit()


