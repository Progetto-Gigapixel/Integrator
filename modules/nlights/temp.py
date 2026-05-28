import cupy as cp
import numpy as np

def check_gpu_memory_info():
    """Comprehensive GPU memory information"""
    
    print("=== GPU Device Information ===")
    device = cp.cuda.Device()
    print(f"Device ID: {device.id}")
    print(f"Device Name: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
    
    print("\n=== Total GPU Memory ===")
    # Method 1: Get total memory from device properties
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    total_memory_bytes = props['totalGlobalMem']
    total_memory_gb = total_memory_bytes / (1024**3)
    print(f"Total GPU Memory: {total_memory_gb:.2f} GB ({total_memory_bytes:,} bytes)")
    
    print("\n=== Current Memory Usage ===")
    # Method 2: Get current memory info
    meminfo = cp.cuda.runtime.memGetInfo()
    free_memory = meminfo[0]  # Free memory in bytes
    total_memory = meminfo[1]  # Total memory in bytes
    used_memory = total_memory - free_memory
    
    print(f"Free Memory:  {free_memory / (1024**3):.2f} GB ({free_memory:,} bytes)")
    print(f"Used Memory:  {used_memory / (1024**3):.2f} GB ({used_memory:,} bytes)")
    print(f"Total Memory: {total_memory / (1024**3):.2f} GB ({total_memory:,} bytes)")
    print(f"Memory Usage: {(used_memory/total_memory)*100:.1f}%")
    
    print("\n=== CuPy Memory Pool Information ===")
    # Method 3: CuPy memory pool statistics
    mempool = cp.get_default_memory_pool()
    print(f"Used bytes:     {mempool.used_bytes():,} bytes ({mempool.used_bytes()/(1024**3):.3f} GB)")
    print(f"Total bytes:    {mempool.total_bytes():,} bytes ({mempool.total_bytes()/(1024**3):.3f} GB)")
    print(f"Free blocks:    {mempool.n_free_blocks()}")
    
    # Pinned memory pool
    pinned_mempool = cp.get_default_pinned_memory_pool()
    print(f"Pinned used:    {pinned_mempool.used_bytes():,} bytes ({pinned_mempool.used_bytes()/(1024**3):.3f} GB)")
    print(f"Pinned total:   {pinned_mempool.total_bytes():,} bytes ({pinned_mempool.total_bytes()/(1024**3):.3f} GB)")

def simple_memory_check():
    """Simple memory check"""
    meminfo = cp.cuda.runtime.memGetInfo()
    free_gb = meminfo[0] / (1024**3)
    total_gb = meminfo[1] / (1024**3)
    used_gb = total_gb - free_gb
    
    print(f"GPU Memory: {used_gb:.2f}/{total_gb:.2f} GB ({used_gb/total_gb*100:.1f}% used)")
    return free_gb, total_gb, used_gb

def monitor_memory_during_operation():
    """Monitor memory usage during operations"""
    print("=== Memory Usage During Operations ===")
    
    # Check memory before allocation
    print("\n1. Before allocation:")
    simple_memory_check()
    
    # Allocate some memory
    print("\n2. Allocating 1GB of data...")
    size = 1024**3 // 4  # 1GB of float32 data
    array1 = cp.ones(size, dtype=cp.float32)
    simple_memory_check()
    
    # Allocate more
    print("\n3. Allocating another 500MB...")
    size2 = (512 * 1024**2) // 4  # 512MB of float32 data
    array2 = cp.ones(size2, dtype=cp.float32)
    simple_memory_check()
    
    # Free one array
    print("\n4. Freeing first array...")
    del array1
    simple_memory_check()
    
    # Clear memory pool
    print("\n5. After clearing memory pool:")
    del array2
    cp.get_default_memory_pool().free_all_blocks()
    simple_memory_check()

def get_max_allocatable_memory():
    """Find maximum allocatable memory chunk"""
    meminfo = cp.cuda.runtime.memGetInfo()
    free_memory = meminfo[0]
    
    # Try to allocate increasingly large chunks
    max_chunk = 0
    test_size = free_memory // 2
    
    print(f"Testing maximum allocatable chunk (Free memory: {free_memory/(1024**3):.2f} GB)")
    
    while test_size > 1024**2:  # Stop at 1MB minimum
        try:
            test_array = cp.ones(test_size // 4, dtype=cp.float32)  # float32 = 4 bytes
            del test_array
            cp.get_default_memory_pool().free_all_blocks()
            max_chunk = test_size
            print(f"✓ Successfully allocated {test_size/(1024**3):.2f} GB")
            break
        except cp.cuda.memory.OutOfMemoryError:
            test_size = test_size // 2
            print(f"✗ Failed to allocate {test_size*2/(1024**3):.2f} GB, trying {test_size/(1024**3):.2f} GB")
    
    return max_chunk

# Quick utility functions
def print_memory_summary():
    """One-liner memory summary"""
    meminfo = cp.cuda.runtime.memGetInfo()
    free_gb = meminfo[0] / (1024**3)
    total_gb = meminfo[1] / (1024**3)
    used_gb = total_gb - free_gb
    print(f"GPU: {used_gb:.1f}/{total_gb:.1f} GB ({used_gb/total_gb*100:.0f}% used, {free_gb:.1f} GB free)")

def get_memory_usage_dict():
    """Return memory info as dictionary"""
    meminfo = cp.cuda.runtime.memGetInfo()
    mempool = cp.get_default_memory_pool()
    
    return {
        'free_bytes': meminfo[0],
        'total_bytes': meminfo[1],
        'used_bytes': meminfo[1] - meminfo[0],
        'pool_used_bytes': mempool.used_bytes(),
        'pool_total_bytes': mempool.total_bytes(),
        'free_gb': meminfo[0] / (1024**3),
        'total_gb': meminfo[1] / (1024**3),
        'used_gb': (meminfo[1] - meminfo[0]) / (1024**3),
        'usage_percent': ((meminfo[1] - meminfo[0]) / meminfo[1]) * 100
    }

if __name__ == "__main__":
    try:
        print("CuPy GPU Memory Information")
        print("=" * 50)
        
        # Basic checks
        check_gpu_memory_info()
        
        print("\n" + "=" * 50)
        
        # Memory monitoring demo
        monitor_memory_during_operation()
        
        print("\n" + "=" * 50)
        
        # Find max allocatable
        max_chunk = get_max_allocatable_memory()
        print(f"\nMaximum allocatable chunk: {max_chunk/(1024**3):.2f} GB")
        
        print("\n" + "=" * 50)
        
        # Quick summary
        print("\nQuick summary:")
        print_memory_summary()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a CUDA-capable GPU and CuPy installed correctly.")