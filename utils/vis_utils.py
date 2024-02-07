import trimesh
import numpy as np
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
import pyrender
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
import os

display = Display(visible=0, size=(1024, 768))
display.start()


def vis_recon(path, data='shapenet', num_view=6):
    if data == 'shapenet':
        save_path = path.replace('Meshes', 'Render').replace('.ply', '.png')
    else:
        raise NotImplementedError
    
    dir_name = os.path.dirname(save_path)
    os.makedirs(dir_name, exist_ok=True)

    mesh_recon = trimesh.load(path)
    mesh = pyrender.Mesh.from_trimesh(mesh_recon)

    # Set up scene and renderer
    scene = pyrender.Scene()
    node = scene.add(mesh)

    # Camera parameters
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 2],
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Headless rendering
    renderer = pyrender.OffscreenRenderer(512, 512)

    fig, axes = plt.subplots(1, num_view, figsize=(num_view*15, 15))
    for i, angle in enumerate(np.linspace(0, 360, num_view+1)):
        if i == num_view:
            break
        ax = axes[i]
        # Set up the rotation matrix
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(angle), [0, 1, 0]
        )
        
        # Update mesh pose using the node
        scene.set_pose(node, pose=rotation_matrix)
        
        # Render the image
        color, depth = renderer.render(scene)

        # Save the rendering
        ax.imshow(color)
        #ax.set_title(f'Azimuth {angle}')
        ax.axis('off')
        #plt.savefig(f'rendered_images/view_{i}.png')

    # Release resources
    #plt.show()
    fig.savefig(save_path, bbox_inches='tight')
    renderer.delete()