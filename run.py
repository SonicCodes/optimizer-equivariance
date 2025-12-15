import torch
from PIL import Image
import numpy as np


class NeuralField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.layers(x)


def prepare_data():
    image_path = "/Users/rami/Downloads/691604.jpeg"
    image = Image.open(image_path)
    image = image.resize((100, 100))
    image = image.convert("RGB")
    image = np.array(image)
    image = image.transpose(1, 2, 0)
    image = image.reshape(1, -1)
    image = torch.from_numpy(image).float()
    image = image / 255.0
    image = image * 2 - 1
    xy_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)), dim=-1)
    return xy_grid.view(100*100, 2), image.view(100*100, 3)


# ============================================================
# Per-parameter rotation matrices
# ============================================================

def make_rotation_for_shape(shape, device, mode='random', seed=None):
    """
    Create rotation matrix appropriate for a parameter's shape.
    
    For weight matrix (out, in): rotate the flattened vector
    For bias (out,): rotate the vector
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dim = np.prod(shape)
    
    if mode == 'random':
        Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
        return Q
    elif mode == 'block45':
        R = torch.eye(dim, device=device)
        c, s = np.cos(np.pi/4), np.sin(np.pi/4)
        for i in range(0, dim - 1, 2):
            R[i, i] = c
            R[i, i+1] = -s
            R[i+1, i] = s
            R[i+1, i+1] = c
        return R
    else:  # identity
        return torch.eye(dim, device=device)


def create_per_param_rotations(model, device, mode='random', seed=42):
    """Create a rotation matrix for each parameter in the model"""
    rotations = []
    for i, p in enumerate(model.parameters()):
        R = make_rotation_for_shape(p.shape, device, mode, seed=seed+i)
        rotations.append(R)
    return rotations


# ============================================================
# Normal training
# ============================================================

def train_normal(model, xy_grid, image, optimizer_class, steps=1000, lr=0.001, **opt_kwargs):
    optimizer = optimizer_class(model.parameters(), lr=lr, **opt_kwargs)
    best_loss = float('inf')
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xy_grid), image)
        loss.backward()
        optimizer.step()
        best_loss = min(best_loss, loss.item())
    return best_loss


# ============================================================
# Rotated training (per-parameter rotation)
# ============================================================

def train_rotated(model, xy_grid, image, rotations, optimizer_class, steps=1000, lr=0.001, **opt_kwargs):
    """
    Each parameter p gets its own rotation R:
        p_tilde = R⁻¹ @ flatten(p)
        p = reshape(R @ p_tilde)
    
    Optimizer works on p_tilde (rotated space)
    """
    # Create rotated parameters
    theta_tildes = []
    for p, R in zip(model.parameters(), rotations):
        R_inv = R.T
        p_tilde = torch.nn.Parameter(R_inv @ p.data.view(-1))
        theta_tildes.append(p_tilde)
    
    # Optimizer on rotated params
    optimizer = optimizer_class(theta_tildes, lr=lr, **opt_kwargs)
    
    best_loss = float('inf')
    
    for _ in range(steps):
        # Map rotated -> original params
        for p, p_tilde, R in zip(model.parameters(), theta_tildes, rotations):
            p.data = (R @ p_tilde).view(p.shape)
        
        # Forward & backward
        model.zero_grad()
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xy_grid), image)
        loss.backward()
        best_loss = min(best_loss, loss.item())
        
        # Set gradients on rotated params: ∇L̃ = R⁻¹ @ ∇L
        for p, p_tilde, R in zip(model.parameters(), theta_tildes, rotations):
            R_inv = R.T
            p_tilde.grad = R_inv @ p.grad.view(-1)
        
        optimizer.step()
    
    return best_loss


# ============================================================
# LR Sweep
# ============================================================

def sweep_lr(train_fn, lrs):
    best_loss = float('inf')
    best_lr = None
    for lr in lrs:
        loss = train_fn(lr)
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
    return best_loss, best_lr


# ============================================================
# Main
# ============================================================

def main():
    device = "mps"
    xy_grid, image = prepare_data()
    xy_grid = xy_grid.to(device)
    image = image.to(device)
    
    lrs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    steps = 10_000
    
    rotation_modes = ['identity', 'block45', 'random']
    
    print("="*70)
    print("Per-Parameter Rotation: Adam vs SGD")
    print(f"LRs tested: {lrs}")
    print("="*70)
    
    results = {}
    
    for mode in rotation_modes:
        print(f"\nTesting rotation mode: {mode}...")
        
        # Adam
        def train_adam(lr):
            torch.manual_seed(42)
            model = NeuralField().to(device)
            if mode == 'identity':
                return train_normal(model, xy_grid, image, torch.optim.Adam, steps, lr)
            else:
                rotations = create_per_param_rotations(model, "cpu", mode, seed=42)
                rotations = [rotation.to(device) for rotation in rotations]
                return train_rotated(model, xy_grid, image, rotations, torch.optim.Adam, steps, lr)
        
        # SGD
        def train_sgd(lr):
            torch.manual_seed(42)
            model = NeuralField().to(device)
            if mode == 'identity':
                return train_normal(model, xy_grid, image, torch.optim.SGD, steps, lr, momentum=0.9)
            else:
                rotations = create_per_param_rotations(model, "cpu", mode, seed=42)
                rotations = [rotation.to(device) for rotation in rotations]
                return train_rotated(model, xy_grid, image, rotations, torch.optim.SGD, steps, lr, momentum=0.9)
        
        adam_loss, adam_lr = sweep_lr(train_adam, lrs)
        sgd_loss, sgd_lr = sweep_lr(train_sgd, lrs)
        
        results[mode] = {
            'adam': (adam_loss, adam_lr),
            'sgd': (sgd_loss, sgd_lr)
        }
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS (best loss @ best LR)")
    print("="*70)
    print(f"\n{'Rotation':<15} {'Adam':<25} {'SGD':<25}")
    print("-"*70)
    
    for mode, res in results.items():
        adam_loss, adam_lr = res['adam']
        sgd_loss, sgd_lr = res['sgd']
        print(f"{mode:<15} {adam_loss:.6f} (lr={adam_lr})    {sgd_loss:.6f} (lr={sgd_lr})")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS (% change from identity)")
    print("="*70)
    
    adam_base = results['identity']['adam'][0]
    sgd_base = results['identity']['sgd'][0]
    
    for mode in ['block45', 'random']:
        adam_rot = results[mode]['adam'][0]
        sgd_rot = results[mode]['sgd'][0]
        
        adam_change = (adam_rot - adam_base) / adam_base * 100
        sgd_change = (sgd_rot - sgd_base) / sgd_base * 100
        
        print(f"\n{mode}:")
        print(f"  Adam: {adam_change:+.1f}%")
        print(f"  SGD:  {sgd_change:+.1f}%")
    
    print("\n" + "="*70)
    print("EXPECTED:")
    print("  • SGD: ~0% change (rotation-equivariant)")
    print("  • Adam: may change (coordinate-dependent)")
    print("="*70)


if __name__ == "__main__":
    main()
