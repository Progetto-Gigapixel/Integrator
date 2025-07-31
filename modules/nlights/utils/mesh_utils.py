import struct
import numpy as np



def write_surface_mesh(mesh_out, file_path, encoding='binary'):
    """
    Write a surface mesh to a PLY file.

    Parameters
    ----------
    mesh_out : dict
        A dictionary containing:
            - 'nodes': ndarray of shape (num_nodes, 3), the vertices of the mesh.
            - 'faces': ndarray of shape (num_faces, 3), the triangular faces of the mesh.
    file_path : str
        Path to the output PLY file.
    encoding : str, optional
        Encoding format for the PLY file ('binary' or 'ascii'). Default is 'binary'.

    Returns
    -------
    None
    """
    nodes = mesh_out['nodes']
    faces = mesh_out['faces']

    if encoding not in ['binary', 'ascii']:
        raise ValueError("Encoding must be 'binary' or 'ascii'.")

    with open(file_path, 'wb' if encoding == 'binary' else 'w') as f:
        # Write PLY header
        f.write("ply\n".encode() if encoding == 'binary' else "ply\n")
        f.write(f"format {encoding}_little_endian 1.0\n".encode() if encoding == 'binary' else f"format {encoding}_little_endian 1.0\n")
        f.write(f"element vertex {len(nodes)}\n".encode() if encoding == 'binary' else f"element vertex {len(nodes)}\n")
        f.write("property float x\n".encode() if encoding == 'binary' else "property float x\n")
        f.write("property float y\n".encode() if encoding == 'binary' else "property float y\n")
        f.write("property float z\n".encode() if encoding == 'binary' else "property float z\n")
        f.write(f"element face {len(faces)}\n".encode() if encoding == 'binary' else f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n".encode() if encoding == 'binary' else "property list uchar int vertex_indices\n")
        f.write("end_header\n".encode() if encoding == 'binary' else "end_header\n")

        # Write vertices
        if encoding == 'binary':
            for node in nodes:
                f.write(np.array(node, dtype=np.float32).tobytes())
        else:
            for node in nodes:
                f.write(f"{node[0]} {node[1]} {node[2]}\n")

        # Write faces
        if encoding == 'binary':
            for face in faces:
                f.write(np.array([3, *face], dtype=np.uint8).tobytes())
        else:
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")



def write_stl(faces,vertices, file_path):
    """
    Write a surface mesh to an STL file in binary format using only face indices.

    Parameters
    ----------
    faces : ndarray
        Array of shape (num_faces, 3) containing the indices of the vertices for each triangular face.
    file_path : str
        Path to the output STL file.

    Returns
    -------
    None
    """
    with open(file_path, 'wb') as f:
        # Write 80-byte header
        header = b'Binary STL Writer' + b' ' * (80 - len('Binary STL Writer'))
        f.write(header)

        # Write number of triangles
        num_faces = faces.shape[0]
        f.write(struct.pack('<I', num_faces))

        # Write each triangle
        for face in faces:
            # Get the vertices of the triangle
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Compute the normal vector (cross product of two edges)
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else np.zeros(3)

            # Write normal vector (3 floats)
            f.write(struct.pack('<3f', *normal))

            # Write vertices (3 vertices, each with 3 floats)
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))

            # Write attribute byte count (2 bytes, set to 0)
            f.write(struct.pack('<H', 0))







