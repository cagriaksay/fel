"""
FEL-CA Simulator (Floating-Point Implementation)

Core implementation of the Flux Equality Law cellular automaton with
Stream-Cancel-Rotate (SCR) rule.
"""

import torch
import numpy as np
from typing import Optional, Tuple


class FELSimulator:
    """
    Deterministic Flux-Equality Cellular Automaton (FEL-CA).
    
    Implements the Stream-Cancel-Rotate (SCR) rule:
    1. Stream: Flux propagates weighted by normalized L1 components
    2. Cancel: Opposing streams cancel to form twist vector T
    3. Rotate: Surviving flux rotates around T by angle θ = κ|T|
    
    Args:
        N: Lattice size (N×N×N cubic grid)
        device: 'cuda', 'mps', or 'cpu'
        rotation_rate: Rotation coefficient κ (radians per unit twist)
        dtype: torch.float32 or torch.float64
    """
    
    def __init__(
        self,
        N: int = 128,
        device: str = 'cuda',
        rotation_rate: float = 0.01,
        dtype: torch.dtype = torch.float32
    ):
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        
        self.N = N
        self.device = device
        self.dtype = dtype
        self.rotation_rate = rotation_rate
        
        # Double-buffered flux field: (N, N, N, 3) for (Fx, Fy, Fz)
        self.F_current = torch.zeros((N, N, N, 3), device=device, dtype=dtype)
        self.F_next = torch.zeros((N, N, N, 3), device=device, dtype=dtype)
        
        # Build rotation lookup table (0.1° resolution)
        self._build_rotation_lut()
        
        self.t = 0  # Current timestep
        
    def _build_rotation_lut(self):
        """Precompute Rodrigues rotation coefficients (0.1° resolution)."""
        import math
        
        # 0.1° resolution = 3600 entries per 360°
        self.angle_resolution = 0.1 * math.pi / 180.0
        self.num_angles = 3600
        
        # Precompute cos(θ), sin(θ), 1-cos(θ) for each angle
        angles = torch.arange(self.num_angles, device=self.device, dtype=self.dtype) * self.angle_resolution
        self.rot_cos = torch.cos(angles)
        self.rot_sin = torch.sin(angles)
        self.rot_one_minus_cos = 1.0 - self.rot_cos
        
    def step(self):
        """
        Execute one SCR step:
        1. Cancel opposing streams from neighbors → rotate around axis
        2. Stream rotated flux weighted by normalized components
        """
        # Step 1: Cancel and Rotate (modifies F_current in-place)
        self._cancel_and_rotate_neighbors()
        
        # Step 2: Stream with normalized L1 weights
        self.F_next.zero_()
        for axis in range(3):
            self._stream_weighted(self.F_current, axis, axis)
        
        # Swap buffers
        self.F_current, self.F_next = self.F_next, self.F_current
        self.t += 1
        
    def _cancel_and_rotate_neighbors(self):
        """
        Cancel opposing streams from neighbors and rotate around combined twist axis.
        
        For each axis, computes cancellation amount c_a = min(I_{+a}, I_{-a}).
        Forms twist vector T = (c_x, c_y, c_z) and rotates F once around T̂ by angle θ = κ|T|.
        """
        # Get flux from all 6 neighbors
        # Note: Array is (Z, Y, X, component) so dims are (0=z, 1=y, 2=x)
        F_from_px = torch.roll(self.F_current, shifts=-1, dims=0)  # +x neighbor (dim 0)
        F_from_nx = torch.roll(self.F_current, shifts=+1, dims=0)  # -x neighbor
        F_from_py = torch.roll(self.F_current, shifts=-1, dims=1)  # +y neighbor (dim 1)
        F_from_ny = torch.roll(self.F_current, shifts=+1, dims=1)  # -y neighbor
        F_from_pz = torch.roll(self.F_current, shifts=-1, dims=2)  # +z neighbor (dim 2)
        F_from_nz = torch.roll(self.F_current, shifts=+1, dims=2)  # -z neighbor
        
        eps = 1e-12
        
        # Unit axis vectors
        ex = torch.tensor([1., 0., 0.], device=self.device, dtype=self.dtype).view(1, 1, 1, 3)
        ey = torch.tensor([0., 1., 0.], device=self.device, dtype=self.dtype).view(1, 1, 1, 3)
        ez = torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype).view(1, 1, 1, 3)
        
        # Calculate cancellation amounts for each axis
        # Project neighbor fluxes onto each axis
        ap_x = torch.sum(F_from_px * ex, dim=-1)  # +x projection
        am_x = torch.sum(F_from_nx * ex, dim=-1)  # -x projection
        ap_y = torch.sum(F_from_py * ey, dim=-1)  # +y projection
        am_y = torch.sum(F_from_ny * ey, dim=-1)  # -y projection
        ap_z = torch.sum(F_from_pz * ez, dim=-1)  # +z projection
        am_z = torch.sum(F_from_nz * ez, dim=-1)  # -z projection
        
        # Keep only opposing streams: ap>0 (forward), am<0 (backward)
        ap_x_pos = torch.clamp(ap_x, min=0.0)
        am_x_neg = -torch.clamp(am_x, max=0.0)
        ap_y_pos = torch.clamp(ap_y, min=0.0)
        am_y_neg = -torch.clamp(am_y, max=0.0)
        ap_z_pos = torch.clamp(ap_z, min=0.0)
        am_z_neg = -torch.clamp(am_z, max=0.0)
        
        # Cancellation amounts: c_a = min(I_{+a}, I_{-a})
        c_x = torch.minimum(ap_x_pos, am_x_neg)
        c_y = torch.minimum(ap_y_pos, am_y_neg)
        c_z = torch.minimum(ap_z_pos, am_z_neg)
        
        # Form twist vector T = (c_x, c_y, c_z) = c_x * ex + c_y * ey + c_z * ez
        # Since ex=(1,0,0), ey=(0,1,0), ez=(0,0,1), we have T = (c_x, c_y, c_z)
        T = torch.stack([c_x, c_y, c_z], dim=-1)
        
        # Twist magnitude |T|
        T_mag = torch.sqrt(torch.sum(T**2, dim=-1) + eps)
        
        # Rotation angle = κ × |T|
        rotation_angle = T_mag * self.rotation_rate
        
        # Normalized twist axis T̂ = T / |T|
        T_hat = T / (T_mag.unsqueeze(-1) + eps)
        
        # Only rotate where there's cancellation (T_mag > eps)
        has_cancel = T_mag > eps
        
        # Skip rotation when κ=0 (cancellation computation still happens)
        if self.rotation_rate == 0.0:
            return
        
        if has_cancel.any():
            # Quantize angle to LUT index
            angle_idx = (rotation_angle / self.angle_resolution).round().long() % self.num_angles
            angle_idx = torch.clamp(angle_idx, 0, self.num_angles - 1)
            
            # Lookup cos/sin
            cos_theta = self.rot_cos[angle_idx].unsqueeze(-1)
            sin_theta = self.rot_sin[angle_idx].unsqueeze(-1)
            one_minus_cos = self.rot_one_minus_cos[angle_idx].unsqueeze(-1)
            
            # Rodrigues rotation around T̂
            # v' = v cos(θ) + (T̂ × v) sin(θ) + T̂(T̂·v)(1-cos(θ))
            T_dot_v = torch.sum(T_hat * self.F_current, dim=-1, keepdim=True)
            
            # T̂ × v cross product
            T_cross_v = torch.cross(T_hat, self.F_current, dim=-1)
            
            # Apply rotation
            F_rotated = (self.F_current * cos_theta +
                        T_cross_v * sin_theta +
                        T_hat * T_dot_v * one_minus_cos)
            
            # Update only where cancellation occurred
            mask = has_cancel.unsqueeze(-1)
            self.F_current = torch.where(mask, F_rotated, self.F_current)
        
    def _stream_weighted(self, F_weight: torch.Tensor, axis: int, component: int):
        """
        Stream flux weighted by normalized L1 components.
        
        Args:
            F_weight: Flux field to use for weights (pre-rotation)
            axis: Spatial dimension to stream along (0=x, 1=y, 2=z)
            component: Flux component for weight (0=Fx, 1=Fy, 2=Fz)
        """
        # L1 norm for normalization
        L1 = torch.sum(torch.abs(F_weight), dim=-1)  # (N, N, N)
        
        # Normalized weight: |F_component| / L1
        weight = torch.where(
            L1 > 1e-10,
            torch.abs(F_weight[..., component]) / L1,
            torch.zeros_like(L1)
        )
        
        # Split into positive/negative directions
        direction = F_weight[..., component]
        weight_pos = torch.where(direction > 0, weight, torch.zeros_like(weight))
        weight_neg = torch.where(direction < 0, weight, torch.zeros_like(weight))
        
        # Stream all three flux components weighted by this axis
        for c in range(3):
            flux_c = self.F_current[..., c]  # Use rotated flux
            
            # Positive direction: +axis
            weighted_pos = flux_c * weight_pos
            self.F_next[..., c] += torch.roll(weighted_pos, shifts=1, dims=axis)
            
            # Negative direction: -axis
            weighted_neg = flux_c * weight_neg
            self.F_next[..., c] += torch.roll(weighted_neg, shifts=-1, dims=axis)
            
    # ============================================================================
    # Initialization Methods
    # ============================================================================
    
    def init_sine_wave(
        self,
        wavelength: float = 32.0,
        direction: str = 'x',
        amplitude: float = 1.0,
        polarization: str = 'y'
    ):
        """
        Initialize with sinusoidal plane wave.
        
        Args:
            wavelength: Wavelength in voxels (λ/Δx)
            direction: Propagation direction ('x', 'y', or 'z')
            amplitude: Wave amplitude
            polarization: Flux direction ('x', 'y', or 'z')
        """
        k = 2 * np.pi / wavelength
        
        # Create coordinate grid
        coords = torch.arange(self.N, device=self.device, dtype=self.dtype)
        
        # Select propagation axis
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        prop_axis = axis_map[direction.lower()]
        pol_axis = axis_map[polarization.lower()]
        
        # Create meshgrid and extract propagation coordinate
        if prop_axis == 0:
            X = coords.view(-1, 1, 1).expand(self.N, self.N, self.N)
        elif prop_axis == 1:
            X = coords.view(1, -1, 1).expand(self.N, self.N, self.N)
        else:  # prop_axis == 2
            X = coords.view(1, 1, -1).expand(self.N, self.N, self.N)
        
        # Sine wave
        sine_wave = amplitude * torch.sin(k * X)
        
        # Assign to flux component
        self.F_current.zero_()
        self.F_current[..., pol_axis] = sine_wave
        
    def init_two_sources(
        self,
        positions: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        amplitude: float = 1.0,
        phase_offset: float = 0.0,
        width: float = 3.0
    ):
        """
        Initialize with two Gaussian point sources.
        
        Args:
            positions: ((x1,y1,z1), (x2,y2,z2)) source positions
            amplitude: Source amplitude
            phase_offset: Phase difference in radians
            width: Gaussian width in voxels
        """
        self.F_current.zero_()
        
        # Create coordinate grids
        x = torch.arange(self.N, device=self.device, dtype=self.dtype)
        y = torch.arange(self.N, device=self.device, dtype=self.dtype)
        z = torch.arange(self.N, device=self.device, dtype=self.dtype)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        for idx, (px, py, pz) in enumerate(positions):
            # Distance from source
            dist_sq = (X - px)**2 + (Y - py)**2 + (Z - pz)**2
            
            # Gaussian envelope
            envelope = amplitude * torch.exp(-dist_sq / (2 * width**2))
            
            # Apply phase offset to second source
            if idx == 1:
                envelope *= torch.cos(torch.tensor(phase_offset, device=self.device))
            
            # Add to y-component (arbitrary choice)
            self.F_current[..., 1] += envelope
            
    # ============================================================================
    # Analysis Methods
    # ============================================================================
    
    def get_magnitude(self) -> torch.Tensor:
        """Get flux magnitude |F| at each voxel."""
        return torch.sqrt(torch.sum(self.F_current**2, dim=-1))
    
    def get_energy(self) -> float:
        """Compute total energy E = Σ|F|²."""
        return torch.sum(self.F_current**2).item()
    
    def get_flux(self) -> torch.Tensor:
        """Get current flux field (N,N,N,3)."""
        return self.F_current.clone()
    
    def compute_inbound_outbound_flux(
        self,
        boundary_faces: torch.Tensor,
        normals: torch.Tensor
    ) -> tuple:
        """
        Compute SEPARATE inbound and outbound flux vectors through boundary.
        
        Returns both as vectors so they can be compared across timesteps for FEL.
        The FEL test is: Inbound(t) should equal Outbound(t+1)
        
        Args:
            boundary_faces: (M, 3) indices of faces on boundary
            normals: (M, 3) outward normal vectors (must be axis-aligned unit vectors)
            
        Returns:
            (inbound_vector, outbound_vector) - both as (3,) tensors
        """
        inbound = torch.zeros(3, device=self.device, dtype=self.dtype)
        outbound = torch.zeros(3, device=self.device, dtype=self.dtype)
        
        for i in range(boundary_faces.shape[0]):
            face = boundary_faces[i]
            normal = normals[i]
            
            x, y, z = int(face[0].item()), int(face[1].item()), int(face[2].item())
            F_here = self.F_current[x, y, z]  # Flux at this voxel
            n = normal  # Outward normal
            
            # Determine which axis and direction
            axis_idx = torch.argmax(torch.abs(n)).item()  # 0=x, 1=y, 2=z
            direction = n[axis_idx].item()  # +1 or -1
            
            # Get neighbor in the direction of the normal
            if axis_idx == 0:  # X axis
                nx, ny, nz = (x + int(direction)) % self.N, y, z
            elif axis_idx == 1:  # Y axis
                nx, ny, nz = x, (y + int(direction)) % self.N, z
            else:  # Z axis
                nx, ny, nz = x, y, (z + int(direction)) % self.N
            
            F_neighbor = self.F_current[nx, ny, nz]  # Flux at neighbor
            
            # Compute streaming weights
            L1_here = torch.sum(torch.abs(F_here)).item()
            L1_neighbor = torch.sum(torch.abs(F_neighbor)).item()
            L1_here = max(L1_here, 1e-10)
            L1_neighbor = max(L1_neighbor, 1e-10)
            
            # Component along this axis
            F_component_here = F_here[axis_idx].item()
            F_component_neighbor = F_neighbor[axis_idx].item()
            
            # OUTBOUND: flux leaving the box
            if direction > 0:
                if F_component_here > 0:
                    weight = abs(F_component_here) / L1_here
                    outbound += F_here * weight
            else:
                if F_component_here < 0:
                    weight = abs(F_component_here) / L1_here
                    outbound += F_here * weight
            
            # INBOUND: flux entering the box
            if direction > 0:
                if F_component_neighbor < 0:
                    weight = abs(F_component_neighbor) / L1_neighbor
                    inbound += F_neighbor * weight
            else:
                if F_component_neighbor > 0:
                    weight = abs(F_component_neighbor) / L1_neighbor
                    inbound += F_neighbor * weight
        
        return inbound, outbound

