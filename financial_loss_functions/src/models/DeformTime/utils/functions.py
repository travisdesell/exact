import torch


def grid_sample1D(tensor, grid):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid.

    Args:
        tensor: (N, C, L_in) tensor
        grid: (N, L_out, 2) tensor in the range of [-1, 1]

    Returns:
        (N, C, L_out) tensor

    """
    b, c, l_in = tensor.shape
    b_, l_out, w_ = grid.shape
    assert b == b_ 
    out = []
    for (t, g) in zip(tensor, grid):
        x_ = 0.5 * (l_in - 1) * (g[:, 0] + 1)
        ix = torch.floor(x_).to(torch.int32).clamp(0, l_in - 2)
        dx = x_ - ix
        out.append((1 - dx) * t[..., ix] + dx * t[..., ix + 1])
    return torch.stack(out, dim=0)

def num_patches(seq_len, patch_len, stride):
    return (seq_len - patch_len) // stride + 1

print(num_patches(96, 7, 4))

