import argparse
import torch
import numpy as np
import triton
import triton.language as tl
import time


def init_problem_once(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Process the lines to remove empty lines and strip spaces
        lines = [line.strip() for line in lines if line.strip()]

        # The first line contains the size
        size = int(lines[0])

        # Initialize matrices a, b, and m
        a = torch.zeros((size, size))
        b = torch.zeros(size)
        m = torch.zeros((size, size))

        # Read matrix 'a' from the file
        for i in range(size):
            a[i] = torch.tensor([float(x) for x in lines[i + 1].split()])

        # Read vector 'b' from the file
        b = torch.tensor([float(x) for x in lines[size + 1].split()])

    return a.float(), b.float(), m.float(), size

def init_per_run(size):
    m = torch.zeros(size, size, dtype=torch.float32)
    return m


def create_matrix(size):
    lamda = -0.01
    coe = np.zeros(2 * size - 1)
    for i in range(size):
        coe_i = 10 * np.exp(lamda * i)
        coe[size - 1 + i] = coe_i
        coe[size - 1 - i] = coe_i

    m = torch.empty(size, size)
    for i in range(size):
        for j in range(size):
            m[i, j] = coe[size - 1 - i + j]

    return m

# Optimized function
def create_matrix_fast(size):
    lamda = -0.01
    indices = np.arange(size)
    coe = 10 * np.exp(lamda * indices)
    coe = np.concatenate([coe[::-1][:-1], coe])

    # Using broadcasting to create the matrix
    i_indices, j_indices = np.ogrid[:size, :size]
    m = coe[size - 1 - i_indices + j_indices]

    return torch.from_numpy(m).float() # Ensure tensor is float32


def forward_sub(a, b, size):
    # Allocate memory on GPU
    m_cuda = torch.zeros((size, size), device='cuda')
    a_cuda = a.clone().to('cuda')
    b_cuda = b.clone().to('cuda')


    BLOCK_SIZE = 1024
    BLOCK_SIZE_X = 16 #64#16
    BLOCK_SIZE_Y = 16 #64#16
    # Begin timing
    start_time = time.time()

    # Forward substitution
    for t in range(size - 1):
        fan1(m_cuda, a_cuda, size, t, BLOCK_SIZE)
        torch.cuda.synchronize()
        fan2(m_cuda, a_cuda, b_cuda, size, size - t, t, BLOCK_SIZE_X, BLOCK_SIZE_Y)
        torch.cuda.synchronize()
        # Check for errors in CUDA operations if needed

    # End timing
    total_kernel_time = time.time() - start_time

    # Copy memory back to CPU
    m = m_cuda.to('cpu')
    a = a_cuda.to('cpu')
    b = b_cuda.to('cpu')

    return m, a, b, total_kernel_time

# Debug version of forward substitution, print out the matrices after each iteration
def forward_sub_debug(a, b, size):
    # Allocate memory on GPU
    m_cuda = torch.zeros((size, size), device='cuda')
    a_cuda = a.clone().to('cuda')
    b_cuda = b.clone().to('cuda')


    BLOCK_SIZE = 1024
    BLOCK_SIZE_X = 16 #64#16
    BLOCK_SIZE_Y = 16 #64#16
    # Begin timing
    print('Matrix a before kernel:')
    print(a)
    print('Matrix m before kernel:')
    print(np.zeros((size, size)))
    start_time = time.time()
    # Forward substitution
    for t in range(size - 1):
        fan1(m_cuda, a_cuda, size, t, BLOCK_SIZE)
        torch.cuda.synchronize()
        # Copy matrix m from device to host to view its current state
        m = m_cuda.to('cpu')
        print(f'Matrix m after iteration {t} (Fan1):')
        print(m)
        fan2(m_cuda, a_cuda, b_cuda, size, size - t, t, BLOCK_SIZE_X, BLOCK_SIZE_Y)
        torch.cuda.synchronize()
        # Check for errors in CUDA operations if needed
        # Copy matrix a from device to host to view its current state
        a = a_cuda.to('cpu')
        print(f'Matrix a after iteration {t} (Fan2):')
        print(a)

    # End timing
    total_kernel_time = time.time() - start_time

    # Copy memory back to CPU
    m = m_cuda.to('cpu')
    a = a_cuda.to('cpu')
    b = b_cuda.to('cpu')

    return m, a, b, total_kernel_time

def back_sub(a, b):
    size = b.size(0)
    final_vec = torch.zeros(size, dtype=a.dtype, device=a.device)

    for i in range(size):
        final_vec[size - i - 1] = b[size - i - 1]
        for j in range(i):
            final_vec[size - i - 1] -= a[size - i - 1, size - j - 1] * final_vec[size - j - 1]
        final_vec[size - i - 1] /= a[size - i - 1, size - i - 1]

    return final_vec

# Optimized back substitution function
def back_sub_fast(a, b):
    size = b.size(0)
    final_vec = b.clone()

    for i in range(size-1, -1, -1):
        # Ensure that the data types match
        a_slice = a[i, i+1:].to(final_vec.dtype)
        final_vec_slice = final_vec[i+1:]

        final_vec[i] -= torch.dot(a_slice, final_vec_slice)
        final_vec[i] /= a[i, i]

    return final_vec

# Fan1 kernel in Triton
@triton.jit
def fan1_kernel(m_cuda_ptr, a_cuda_ptr, Size, t, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    rows = block_start + tl.arange(0, BLOCK_SIZE)

    # Ensure we don't process rows beyond our matrix size
    valid_rows = rows < Size - 1 - t
    rows = rows + t + 1

    # Calculating linear indices for 2D array access
    idx_m_cuda = rows * Size + t
    idx_a_cuda_row = rows * Size + t
    idx_a_cuda_col = t * Size + t

    # Load from memory
    m_cuda = tl.load(m_cuda_ptr + idx_m_cuda, mask=valid_rows)
    a_cuda_row = tl.load(a_cuda_ptr + idx_a_cuda_row, mask=valid_rows)
    a_cuda_col = tl.load(a_cuda_ptr + idx_a_cuda_col, mask=t < Size)

    # Perform the division
    m_cuda = a_cuda_row / a_cuda_col

    # Store to memory
    tl.store(m_cuda_ptr + idx_m_cuda, m_cuda, mask=valid_rows)


def fan1(m_cuda, a_cuda, size, t, BLOCK_SIZE):
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fan1_kernel[grid](m_cuda, a_cuda, size, t, BLOCK_SIZE)

# Fan2 kernel in Triton
@triton.jit
def fan2_kernel(m_cuda_ptr, a_cuda_ptr, b_cuda_ptr, size, j1, t, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    x_start = pid_x * BLOCK_SIZE_X
    y_start = pid_y * BLOCK_SIZE_Y

    x_idx = x_start + tl.arange(0, BLOCK_SIZE_X)
    y_idx = y_start + tl.arange(0, BLOCK_SIZE_Y)

    valid_x = x_idx + t + 1 < size
    valid_y = y_idx + t < size

    x_idx = x_idx + t + 1
    y_idx = y_idx + t

    idx_a_cuda_w = x_idx * size + y_idx
    idx_a_cuda_r = size * t + y_idx
    idx_m_cuda = x_idx * size + t

    a_cuda_w = tl.load(a_cuda_ptr + idx_a_cuda_w, mask=valid_x & valid_y)
    a_cuda_r = tl.load(a_cuda_ptr + idx_a_cuda_r, mask=valid_y)
    m_cuda = tl.load(m_cuda_ptr + idx_m_cuda, mask=valid_x)

    a_cuda_w = a_cuda_w - m_cuda * a_cuda_r

    tl.store(a_cuda_ptr + idx_a_cuda_w, a_cuda_w, mask=valid_x & valid_y)

    conditional_mask = y_idx == t # Takes the role of if statement

    idx_b_cuda_r = t
    idx_m_cuda_conditional = x_idx * size + y_idx
    idx_b_cuda_w = x_idx

    b_cuda_r = tl.load(b_cuda_ptr + idx_b_cuda_r, mask=(t < size), other = 0)
    m_cuda_conditional = tl.load(m_cuda_ptr + idx_m_cuda_conditional, mask=valid_x & valid_y & conditional_mask, other = 0)
    b_cuda_w = tl.load(b_cuda_ptr + idx_b_cuda_w, mask=valid_x & conditional_mask, other = 0)

    b_cuda_w = b_cuda_w - m_cuda_conditional * b_cuda_r
    tl.store(b_cuda_ptr + idx_b_cuda_w, b_cuda_w, mask=valid_x & conditional_mask)


def fan2(m_cuda, a_cuda, b_cuda, size, j1, t, BLOCK_SIZE_X, BLOCK_SIZE_Y):
    grid = ((size + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X, (size + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y)
    fan2_kernel[grid](m_cuda, a_cuda, b_cuda, size, j1, t, BLOCK_SIZE_X, BLOCK_SIZE_Y)

def main():
    parser = argparse.ArgumentParser(description='Gaussian Elimination in PyTorch and Triton')
    parser.add_argument('-s', '--size', type=int, help='Size of the matrix')
    parser.add_argument('-f', '--file', type=str, help='Filename of the input file')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress printing the matrix and result values')
    parser.add_argument('-d', '--debug', type=int, help='Run in debug mode to print out matrices after each iteration')

    args = parser.parse_args()

    verbose = not args.quiet
    size = args.size
    filename = args.file
    debug = args.debug
    print(f"Running with size = {size}, filename = {filename}, verbose = {verbose}, debug = {debug}")
    if size:
        print(f"Create matrix internally in parse, size = {size}")
        a = create_matrix_fast(size) # a = create_matrix(size)
        b = torch.ones(size)
        m = init_per_run(size)
        # m matrix needs to be defined based on your specific needs
    elif filename:
        print(f"Read file from {filename}")
        # Implement file reading and matrix initialization
        a, b, m, size = init_problem_once(filename)
    else:
        print("No size or file specified")
        return
    
    if verbose:
        print('Initial values:')
        print("Matrix a is:\n", a)
        print("Matrix b is:\n", b)
        print("Matrix m is:\n", m)

    if debug:
        print('Debug mode enabled')
        total_time_start = time.time()
        m, a, b, total_kernel_time = forward_sub_debug(a, b, size)
        total_time = time.time() - total_time_start
    else:
        total_time_start = time.time()
        m, a, b, total_kernel_time = forward_sub(a, b, size)
        total_time = time.time() - total_time_start

    # Call to forward and backward substitution and other operations will go here
    

    if verbose:
        print("Matrix a is:\n", a)
        print("Matrix b is:\n", b)
        print("Matrix m is:\n", m)

    final_vec = back_sub_fast(a, b)# back_sub(a, b)
    if verbose:
        print("Final solution is:\n", final_vec)

    print(f"Total time (including memory transfers): {total_time} sec")
    print(f"Time for CUDA kernels: {total_kernel_time} sec")

if __name__ == "__main__":
    main()

