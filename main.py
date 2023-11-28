import numpy as np
import pygame
from pyopencl import *
from random import randint, random
from os import environ
import time

BG = (0, 0, 0)
FG = (255, 255, 255)

# Easy access mem flags.
WRITE_ONLY = mem_flags.WRITE_ONLY
READ_ONLY = mem_flags.READ_ONLY
COPY_HOST_PTR = mem_flags.COPY_HOST_PTR

# Set default mode for pyopencl to use.
environ["PYOPENCL_CTX"] = "0"

# Opencl context.
context = create_some_context()
context_queue = CommandQueue(context)

# Datatypes
RGB = np.dtype([
    ("r", np.int32),
    ("g", np.int32),
    ("b", np.int32)])

Vector2D = np.dtype([
    ("x", np.float32),
    ("y", np.float32)])

Particle = np.dtype([
    ("position", Vector2D),
    ("velocity", Vector2D),
    ("orbit", Vector2D)])


# Create window.
pygame.init()
window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
window_alpha = window.convert_alpha()
width = window.get_width()
height = window.get_height()

render_buffer = np.zeros((width, height), dtype=RGB)

# GPU Compute Particle function.
__update_particles = Program(context, """
typedef struct rgb {
    int r;
    int g;
    int b;
} RGB;


typedef struct vector2d {
    float x;
    float y;
} Vector2D;


typedef struct particle {
    Vector2D position;
    Vector2D velocity;
    Vector2D orbit;
} Particle;


__kernel void update_particles(
    __global const Particle *particles,
    __global Particle *new_particles,
    __global RGB *surfarray,
    const int width,
    const int height)
{
    int gid = get_global_id(0);

    // Calculate Force Vector.
    Vector2D force;
    force.x = particles[gid].orbit.x - particles[gid].position.x;
    force.y = particles[gid].orbit.y - particles[gid].position.y;

    // Normalise the force vector.
    float mag = sqrt((force.x * force.x) + (force.y * force.y));

    if (mag > 0.0f) {
        force.x = force.x / mag;
        force.y = force.y / mag;
    }

    // New Velocity.
    Vector2D velocity;
    velocity.x = (particles[gid].velocity.x + force.x);
    velocity.y = (particles[gid].velocity.y + force.y);

    // New Position.
    Vector2D position;
    position.x = particles[gid].position.x + particles[gid].velocity.x;
    position.y = particles[gid].position.y + particles[gid].velocity.y;

    // New Particle.
    Particle new_particle = {position, velocity, particles[gid].orbit};
    new_particles[gid] = new_particle;

    // Color pixel if on screen.
    int x = (int) particles[gid].position.x;
    int y = (int) particles[gid].position.y;
    
    if (x < 0 || x >= width) return;
    if (y < 0 || y >= height) return;

    surfarray[(y * width) + x].r = 255;
    surfarray[(y * width) + x].g = 255;
    surfarray[(y * width) + x].b = 255;
}

""").build()
__update_particles = __update_particles.update_particles


def update_particles(particles) -> np.array:
    """Updates the particles, pulling them towards their center."""
    new_particles = np.zeros_like(particles)

    # GPU Buffers.
    particles_gpu = Buffer(context, READ_ONLY | COPY_HOST_PTR, hostbuf=particles)
    new_particles_gpu = Buffer(context, WRITE_ONLY, new_particles.nbytes)
    render_buffer_gpu = Buffer(context, WRITE_ONLY, render_buffer.nbytes)
    
    # Updating,
    __update_particles(context_queue, new_particles.shape, None,
                       particles_gpu, new_particles_gpu,
                       render_buffer_gpu, np.int32(width), np.int32(height))

    # Retrieve new particles and drawn particles.
    enqueue_copy(context_queue, new_particles, new_particles_gpu)
    enqueue_copy(context_queue, render_buffer, render_buffer_gpu)
    return new_particles


# Simulation variables.
center = [width//2, height//2]
n_particles = 100
running = True

frames = "frames.npy"
total_frames = 10 * 60

with open(frames, "w"):
    # Just cleaning the file.
    pass


# Information handling.
now = time.time()
last = now

# Create particles.
particles = np.zeros((n_particles,), dtype=Particle)
for i in range(n_particles):
    particles[i]["position"]["x"] = randint(0, width)
    particles[i]["position"]["y"] = randint(0, height)
    particles[i]["velocity"]["x"] = random() - 0.5
    particles[i]["velocity"]["y"] = random() - 0.5
    particles[i]["orbit"]["x"] = width // 2
    particles[i]["orbit"]["y"] = height // 2

# Pre compute frames.
for i in range(total_frames):
    print(f"{i + 1}/{total_frames}")
    # Update particles.
    particles = update_particles(particles)

    # Change render_buffer to valid 3D array.
    render_buffer_3d = np.zeros((width, height, 3), dtype=np.uint8)
    render_buffer_3d[:,:,0] = render_buffer['r']
    render_buffer_3d[:,:,1] = render_buffer['g']
    render_buffer_3d[:,:,2] = render_buffer['b']
    render_buffer = np.zeros_like(render_buffer)

    with open(frames, "ab") as file:
        np.save(file, render_buffer_3d)
    
    
with open(frames, "rb") as file:
    while True:
        try:
            frame = np.load(file)
            frame = pygame.surfarray.make_surface(frame)
            # Draw particles.
            window.fill(BG)
            window.blit(frame, (0, 0))
            pygame.display.update()
            
        except:
            break

        # Event handling.
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    running = False

pygame.quit()
