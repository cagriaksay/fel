"""
Utility functions for FEL-CA analysis and metrics.
"""

import torch
import numpy as np
from typing import Tuple, Dict


def build_rotation_lut(
    resolution_deg: float = 0.1,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """
    Build rotation lookup table for Rodrigues formula.
    
    Args:
        resolution_deg: Angular resolution in degrees
        device: torch device
        dtype: torch dtype
        
    Returns:
        Dictionary with 'cos', 'sin', 'one_minus_cos', 'resolution', 'num_angles'
    """
    import math
    
    resolution_rad = resolution_deg * math.pi / 180.0
    num_angles = int(360.0 / resolution_deg)
    
    angles = torch.arange(num_angles, device=device, dtype=dtype) * resolution_rad
    
    return {
        'cos': torch.cos(angles),
        'sin': torch.sin(angles),
        'one_minus_cos': 1.0 - torch.cos(angles),
        'resolution': resolution_rad,
        'num_angles': num_angles
    }


def compute_energy(F: torch.Tensor) -> float:
    """
    Compute total energy E = Σ|F|².
    
    Args:
        F: Flux field (N,N,N,3)
        
    Returns:
        Total energy
    """
    return torch.sum(F**2).item()


def compute_helicity(F: torch.Tensor) -> float:
    """
    Compute helicity-like invariant H = Σ F·(∇×F).
    
    Uses finite-difference curl on 3D lattice.
    
    Args:
        F: Flux field (N,N,N,3)
        
    Returns:
        Helicity value
    """
    # Compute curl using finite differences
    # ∇×F = (∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y)
    
    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
    
    # Finite differences (periodic boundaries)
    dFx_dy = torch.roll(Fx, shifts=-1, dims=1) - torch.roll(Fx, shifts=1, dims=1)
    dFx_dz = torch.roll(Fx, shifts=-1, dims=2) - torch.roll(Fx, shifts=1, dims=2)
    
    dFy_dx = torch.roll(Fy, shifts=-1, dims=0) - torch.roll(Fy, shifts=1, dims=0)
    dFy_dz = torch.roll(Fy, shifts=-1, dims=2) - torch.roll(Fy, shifts=1, dims=2)
    
    dFz_dx = torch.roll(Fz, shifts=-1, dims=0) - torch.roll(Fz, shifts=1, dims=0)
    dFz_dy = torch.roll(Fz, shifts=-1, dims=1) - torch.roll(Fz, shifts=1, dims=1)
    
    # Curl components
    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
    
    # Stack curl
    curl = torch.stack([curl_x, curl_y, curl_z], dim=-1)
    
    # H = F · curl
    helicity = torch.sum(F * curl).item()
    
    return helicity / 2.0  # Factor of 2 from finite difference


def spectral_analysis(
    F: torch.Tensor,
    component: int = 1,
    slice_axis: int = 0,
    slice_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform 2D FFT on a slice to analyze spatial spectrum.
    
    Args:
        F: Flux field (N,N,N,3)
        component: Flux component to analyze (0=x, 1=y, 2=z)
        slice_axis: Axis perpendicular to slice (0=x, 1=y, 2=z)
        slice_idx: Index along slice axis
        
    Returns:
        (freqs, power_spectrum) where freqs are in cycles/voxel
    """
    # Extract slice
    if slice_axis == 0:
        slice_data = F[slice_idx, :, :, component]
    elif slice_axis == 1:
        slice_data = F[:, slice_idx, :, component]
    else:  # slice_axis == 2
        slice_data = F[:, :, slice_idx, component]
    
    # Convert to numpy
    slice_np = slice_data.cpu().numpy()
    
    # 2D FFT
    fft_2d = np.fft.fft2(slice_np)
    fft_2d = np.fft.fftshift(fft_2d)
    
    # Power spectrum
    power = np.abs(fft_2d)**2
    
    # Frequency axes
    N = slice_np.shape[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(N))
    
    return freqs, power


def measure_phase_speed(
    trajectory: np.ndarray,
    dt: float = 1.0,
    dx: float = 1.0
) -> Tuple[float, float]:
    """
    Measure phase speed from space-time trajectory.
    
    Uses spectral peak tracking to extract ω and k, then c_phase = ω/k.
    
    Args:
        trajectory: (T, X) space-time data
        dt: Time step
        dx: Spatial step
        
    Returns:
        (c_phase, error_estimate)
    """
    # 2D FFT of space-time
    fft_2d = np.fft.fft2(trajectory)
    fft_2d = np.fft.fftshift(fft_2d)
    
    # Power spectrum
    power = np.abs(fft_2d)**2
    
    # Find peak (exclude DC)
    T, X = trajectory.shape
    center_t, center_x = T // 2, X // 2
    
    # Mask out center ±3 pixels
    mask = np.ones_like(power, dtype=bool)
    mask[center_t-3:center_t+3, center_x-3:center_x+3] = False
    
    # Find peak
    peak_idx = np.unravel_index(np.argmax(power * mask), power.shape)
    
    # Convert to frequencies
    freq_t = np.fft.fftshift(np.fft.fftfreq(T, dt))
    freq_x = np.fft.fftshift(np.fft.fftfreq(X, dx))
    
    omega = 2 * np.pi * freq_t[peak_idx[0]]
    k = 2 * np.pi * freq_x[peak_idx[1]]
    
    # Phase speed
    if abs(k) > 1e-10:
        c_phase = omega / k
    else:
        c_phase = 0.0
    
    # Error estimate (from peak width)
    # Simple estimate: FWHM of peak
    error = 0.1 * c_phase  # 10% placeholder
    
    return abs(c_phase), error


def create_spherical_boundary(
    center: Tuple[int, int, int],
    radius: int,
    N: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a CONNECTED spherical boundary for flux measurement.
    
    Uses a proper surface extraction:
    1. For each voxel, classify as inside (dist < r) or outside (dist >= r)
    2. A voxel is on the boundary if it's outside AND has at least one inside neighbor
    
    This ensures a single-voxel-thick, gap-free boundary.
    
    Args:
        center: (cx, cy, cz) sphere center
        radius: Sphere radius in voxels
        N: Lattice size
        
    Returns:
        (boundary_faces, normals) where:
            boundary_faces: (M, 3) integer coordinates (unique voxels on shell)
            normals: (M, 3) outward unit normals
    """
    cx, cy, cz = center
    
    # Use a bounding box to limit search
    bbox_min = max(0, int(cx - radius - 2))
    bbox_max = min(N, int(cx + radius + 3))
    
    # Step 1: Create distance field for candidate voxels
    candidate_voxels = {}
    for ix in range(bbox_min, bbox_max):
        for iy in range(bbox_min, bbox_max):
            for iz in range(bbox_min, bbox_max):
                dx = ix - cx
                dy = iy - cy
                dz = iz - cz
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Only consider voxels near the boundary
                if abs(dist - radius) <= 1.5:
                    candidate_voxels[(ix, iy, iz)] = dist
    
    # Step 2: Extract boundary surface
    # A voxel is on the boundary if:
    # - It's at distance >= r (outside or on surface)
    # - At least one 6-neighbor is at distance < r (inside)
    faces = []
    normals = []
    
    for (ix, iy, iz), dist in candidate_voxels.items():
        # Check if this voxel is outside (or on) the sphere
        if dist >= radius - 0.5:  # Allow slight tolerance
            # Check 6 neighbors
            neighbors = [
                (ix-1, iy, iz), (ix+1, iy, iz),
                (ix, iy-1, iz), (ix, iy+1, iz),
                (ix, iy, iz-1), (ix, iy, iz+1),
            ]
            
            # Check if any neighbor is inside
            has_inside_neighbor = False
            for (nx, ny, nz) in neighbors:
                if (nx, ny, nz) in candidate_voxels:
                    n_dist = candidate_voxels[(nx, ny, nz)]
                    if n_dist < radius - 0.5:  # Clearly inside
                        has_inside_neighbor = True
                        break
                else:
                    # Compute distance for non-candidate neighbor
                    ndx = nx - cx
                    ndy = ny - cy
                    ndz = nz - cz
                    n_dist = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
                    if n_dist < radius - 0.5:
                        has_inside_neighbor = True
                        break
            
            # Only include boundary voxels
            if has_inside_neighbor:
                faces.append([ix, iy, iz])
                
                # Outward normal (from center)
                dx = ix - cx
                dy = iy - cy
                dz = iz - cz
                norm_val = dist + 1e-10
                normals.append([dx/norm_val, dy/norm_val, dz/norm_val])
    
    return torch.tensor(faces, dtype=torch.long), torch.tensor(normals, dtype=torch.float32)


def create_box_boundary(
    corner1: Tuple[int, int, int],
    corner2: Tuple[int, int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create axis-aligned box boundary measuring flux through actual outward FACES.
    
    For each surface voxel, only include the faces that point OUTWARD:
    - Face voxels (on one boundary plane): 1 face
    - Edge voxels (on two boundary planes): 2 faces (the two outward ones)
    - Corner voxels (on three boundary planes): 3 faces (all three outward ones)
    
    Args:
        corner1: (x1, y1, z1) minimum corner
        corner2: (x2, y2, z2) maximum corner
        
    Returns:
        (boundary_faces, normals) where each voxel appears once per outward face
    """
    x1, y1, z1 = corner1
    x2, y2, z2 = corner2
    
    faces = []
    normals = []
    
    # Iterate over all surface voxels
    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            for z in range(z1, z2+1):
                # Check if this voxel is on the surface
                on_x_min = (x == x1)
                on_x_max = (x == x2)
                on_y_min = (y == y1)
                on_y_max = (y == y2)
                on_z_min = (z == z1)
                on_z_max = (z == z2)
                
                # Skip interior voxels
                if not (on_x_min or on_x_max or on_y_min or on_y_max or on_z_min or on_z_max):
                    continue
                
                # Add outward faces based on which boundary planes this voxel is on
                if on_x_min:
                    faces.append([x, y, z])
                    normals.append([-1, 0, 0])
                
                if on_x_max:
                    faces.append([x, y, z])
                    normals.append([+1, 0, 0])
                
                if on_y_min:
                    faces.append([x, y, z])
                    normals.append([0, -1, 0])
                
                if on_y_max:
                    faces.append([x, y, z])
                    normals.append([0, +1, 0])
                
                if on_z_min:
                    faces.append([x, y, z])
                    normals.append([0, 0, -1])
                
                if on_z_max:
                    faces.append([x, y, z])
                    normals.append([0, 0, +1])
    
    return torch.tensor(faces, dtype=torch.long), torch.tensor(normals, dtype=torch.float32)

