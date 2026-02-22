
import torch
import matplotlib.pyplot as plt
import array_api_compat.torch as xp

from phantoms import primitives, trajectories, generator
from scanner import scanner
from lines import representations, metrics
from tda import diagrams, distances, vectorization


def run_morph_test():
    # Simulation and phantom configuration
    num_steps = 7
    num_lors = 2000

    # TDA configuration parameters
    hom_dim = 1  # Use 0 for clusters, 1 for loops
    method = 'wasserstein'  # Use 'wasserstein' or 'bottleneck'

    projector = scanner.get_scanner(xp, 'cpu')
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[1] // 4  # Set radius to a quarter of the Y dimension for good visibility 
    print(f"Using scanner with image shape: {grid_shape} and radius: {projector.lor_descriptor.scanner.__getattribute__('radius')}")

    # Create a static trajectory to isolate shape changes from motion
    path = trajectories.static_trajectory(
        center=(0, 0, 0),
        steps=num_steps
    )

    # Define the morphing function: 0.0 (sphere) -> 1.0 (cube)
    def morph_modifier(step):
        return {'morph_factor': step / (num_steps - 1)}

    print("Generating morphed phantom frames...")
    # Generate frames using the morphed primitive
    frames = generator.generate_frames(
        primitives.create_morphed_phantom,
        trajectory=path,
        shape_func=morph_modifier,
        shape=grid_shape,
        radius=phantom_radius
    )

    print("Frames generated. Visualizing the phantom sequence...")
    # Visualize the phantom sequence to verify the morphing
    generator.plot_frame_sequence(frames)

    # Initialize storage for topological signatures of each frame
    all_diagrams = []

    # Frame processing loop
    for i, frame in enumerate(frames):
        # Sample LOR endpoints from the current image
        p1, p2 = scanner.generate_lors_from_image(frame, projector, num_lors=num_lors, show=False)

        # Convert 3D endpoints to canonical 6D Plücker coordinates
        plucker_coords = representations.to_canonical_plucker(p1, p2)
        print(f"shape of plucker coords: {plucker_coords.shape}")

        # Compute the pairwise distance matrix using the custom line metric
        dist_matrix = metrics.compute_hybrid_weighted_distance(
            plucker_coords,
            alpha=1,
            beta=1 / projector.lor_descriptor.scanner.__getattribute__('radius')
        )
        #dist_matrix = metrics.compute_euclidean_distance(plucker_coords)

        # Calculate the persistence diagram for the current frame
        dgm = diagrams.compute_persistence(dist_matrix, is_distance_matrix=True)
        #diagrams.plot_persistence_diagram(dgm, title=f"Frame {i + 1} Persistence Diagram")

        all_diagrams.append(dgm)

        print(f"Frame {i + 1}/{num_steps} processed.")

    print("All frames processed. Computing inter-frame distances...")
    # Compute a distance matrix comparing every frame to every other frame
    dist_heatmap = distances.compute_all_pairs_distances(
        all_diagrams,
        method=method,
        hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    # Visualize the inter-frame differences as a topological heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dist_heatmap, cmap='viridis', origin='lower')

    method_name = method.capitalize()
    hom_name = f"H{hom_dim}"

    plt.colorbar(im)
    plt.title(f"{method_name} Distance ({hom_name}) Heatmap – Morph Test")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")

    # Annotate the heatmap cells with numerical distance values
    for i in range(num_steps):
        for j in range(num_steps):
            plt.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                     ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.show()

def run_motion_analysis():
    # Simulation and phantom configuration
    num_steps = 8
    num_lors = 3000

    # TDA configuration parameters
    hom_dim = 1  # Use 0 for clusters, 1 for loops
    method = 'wasserstein'  # Use 'wasserstein' or 'bottleneck'

    # PET scanner geometry and projection settings
    projector = scanner.get_scanner(xp, 'cpu')
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[1] // 8  # Set radius to an eighth of the Y dimension for good visibility
    print(f"Using scanner with image shape: {grid_shape} and radius: {projector.lor_descriptor.scanner.__getattribute__('radius')}")

    # Generate a linear trajectory path for the phantom movement
    path = trajectories.linear_trajectory(
        start=(-grid_shape[0]//2+phantom_radius, -grid_shape[1]//2+phantom_radius, 0),
        end=(grid_shape[0]//2-phantom_radius, grid_shape[1]//2-phantom_radius, 0),
        steps=num_steps
    )

    # Produce the sequence of 3D image frames based on the trajectory
    frames = generator.generate_frames(
        primitives.create_sphere_phantom,
        trajectory=path,
        shape=grid_shape,
        radius=phantom_radius
    )

    generator.plot_frame_sequence(frames)

    all_diagrams = []
    for i, frame in enumerate(frames):
        p1, p2 = scanner.generate_lors_from_image(frame, projector, num_lors=num_lors)
        plucker_coords = representations.to_canonical_plucker(p1, p2)
        dist_matrix = metrics.compute_hybrid_weighted_distance(
            plucker_coords,
            alpha=1,
            beta=1 / projector.lor_descriptor.scanner.__getattribute__('radius')
        )
        dgm = diagrams.compute_persistence(dist_matrix, is_distance_matrix=True)
        all_diagrams.append(dgm)
        print(f"Frame {i + 1}/{num_steps} processed.")

    print("All frames processed. Computing inter-frame distances...")
    # Compute a distance matrix comparing every frame to every other frame
    dist_heatmap = distances.compute_all_pairs_distances(
        all_diagrams,
        method=method,
        hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    # Visualize the inter-frame differences as a topological heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dist_heatmap, cmap='viridis', origin='lower')
    method_name = method.capitalize()
    hom_name = f"H{hom_dim}"
    plt.colorbar(im)
    plt.title(f"{method_name} Distance ({hom_name}) Heatmap")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")

    for i in range(num_steps):
        for j in range(num_steps):
            plt.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                     ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.show()

def run_gating_simulation():
    # Simulation and phantom configuration

    num_steps = 8
    num_lors = 2000

    # TDA and Gating settings
    hom_dim = 1
    method = 'wasserstein'
    img_res = (40,40)

    # PET scanner geometry and projection settings
    projector = scanner.get_scanner(xp, 'cpu')
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[1] // 8  # Set radius to an eighth of the Y dimension for good visibility
    print(f"Using scanner with image shape: {grid_shape} and radius: {projector.lor_descriptor.scanner.__getattribute__('radius')}")

    # Generate a periodic trajectory to simulate breathing (2 full cycles)
    path = trajectories.periodic_trajectory(
        center=(0, 0, 0),
        amplitude=(0, 20 - phantom_radius, 0),
        steps=num_steps,
        frequency=2.0
    )

    # Produce the sequence of 3D image frames
    frames = generator.generate_frames(
        primitives.create_sphere_phantom,
        trajectory=path,
        shape=grid_shape,
        radius=phantom_radius
    )

    # Visualize the ground truth phantom sequence
    generator.plot_frame_sequence(frames, title="Breathing Cycle Ground Truth")

    all_diagrams = []
    for i, frame in enumerate(frames):
        p1, p2 = scanner.generate_lors_from_image(frame, projector, num_lors=num_lors, show=True)
        plucker_coords = representations.to_canonical_plucker(p1, p2)
        dist_matrix = metrics.compute_hybrid_weighted_distance(
            plucker_coords,
            alpha=1,
            beta=1 / projector.lor_descriptor.scanner.__getattribute__('radius')
        )
        dgm = diagrams.compute_persistence(dist_matrix, is_distance_matrix=True)
        all_diagrams.append(dgm)
        print(f"Frame {i + 1}/{num_steps} processed.")

    print("All frames processed. Computing persistence images...")
    # Generate persistence images for vector-based gating/ML
    pers_images = vectorization.get_persistence_images(
        all_diagrams,
        hom_dim=hom_dim,
        resolution=img_res
    )

    print("All frames processed. Computing inter-frame distances...")
    # Compute a distance matrix comparing every frame to every other frame
    dist_heatmap = distances.compute_all_pairs_distances(
        all_diagrams,
        method=method,
        hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    # Visualize the inter-frame differences as a topological heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dist_heatmap, cmap='viridis', origin='lower')
    method_name = method.capitalize()
    hom_name = f"H{hom_dim}"
    plt.colorbar(im)
    plt.title(f"{method_name} Distance ({hom_name}) Heatmap")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")

    for i in range(num_steps):
        for j in range(num_steps):
            plt.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                     ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_morph_test()
    #run_motion_analysis()
    #run_gating_simulation()