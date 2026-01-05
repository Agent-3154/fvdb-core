import torch
import time
import numpy as np
import math
from tqdm import tqdm
from fvdb import GaussianSplat3d
from collections import defaultdict
from gsplat.rendering import rasterization
import matplotlib.pyplot as plt

# NVTX for profiling
from torch.cuda import nvtx
"""
Usage: nsys profile --trace=cuda,osrt python benchmark_rendering.py
"""

if __name__ == "__main__":
    # width, height = 480, 360
    width, height = 640, 480
    # width, height = 800, 600
    tile_size = 16
    width_tiles = int(math.ceil(width / tile_size))
    height_tiles = int(math.ceil(height / tile_size))

    NUM_CAMERAS = [16, 32, 48, 64, 80, 96, 112, 128]
    num_warmup = 10
    num_iterations = 50
    min_radius_2d = 3.0

    device = torch.device("cuda")

    ply_path = "/home/btx0424/lab51/3d-pipeline/r2s-gs/data/grand_tour_dataset/2024-10-01-11-47-44/splats/1148c086/splatfacto/splat.ply"
    # ply_path = "/home/btx0424/lab51/3d-pipeline/r2s-gs/data/grand_tour_dataset/2024-10-01-11-47-44/splats/1407d603/splatfacto/splat.ply"
    gs, metadata = GaussianSplat3d.from_ply(ply_path, device=device)

    w2c = torch.tensor([[ 6.0837e-01,  7.9366e-01, -7.7965e-09,  2.6445e-08],
        [ 1.6030e-01, -1.2287e-01, -9.7939e-01,  7.9346e-09],
        [-7.7730e-01,  5.9583e-01, -2.0197e-01,  6.4092e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device=device)
    
    fx = fy = width * 0.7
    cx = width / 2
    cy = height / 2
    K = torch.tensor([[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]], device=device)
    
    def render_fvdb(w2c: torch.Tensor, K: torch.Tensor):
        images_nhwc, alphas = gs.render_images(
            world_to_camera_matrices=w2c.contiguous(),
            projection_matrices=K.contiguous(),
            image_width=width,
            image_height=height,
            near=0.1,
            far=100.0,
            sh_degree_to_use=3,
            tile_size=16,
            min_radius_2d=min_radius_2d,
            eps_2d=0.3,
        )
        return images_nhwc, alphas
    
    P = (width * height) // 16
    xx = torch.arange(width, device=device)
    yy = torch.arange(height, device=device)

    pixel_coords = torch.stack(torch.meshgrid(yy, xx, indexing='ij'), dim=-1).reshape(-1, 2)
    def render_fvdb_sparse(w2c: torch.Tensor, K: torch.Tensor):
        indices =  torch.rand(w2c.shape[0], width * height, device=device).argsort(dim=1)
        pixels_to_render = pixel_coords[indices[:, :P]].reshape(w2c.shape[0], P, 2)

        images_nhwc, alphas = gs.sparse_render_images(
            pixels_to_render=pixels_to_render.contiguous(), # [C, P, 2]
            world_to_camera_matrices=w2c.contiguous(),
            projection_matrices=K.contiguous(),
            image_width=width,
            image_height=height,
            near=0.1,
            far=100.0,
            sh_degree_to_use=3,
            tile_size=16,
            min_radius_2d=min_radius_2d,
            eps_2d=0.3,
        )
        return images_nhwc, alphas
    
    def render_fvdb_tile(w2c: torch.Tensor, K: torch.Tensor):
        P = (width_tiles * height_tiles) // 4
        xx = torch.arange(width_tiles, device=device, dtype=torch.int32)
        yy = torch.arange(height_tiles, device=device, dtype=torch.int32)
        tile_coords = torch.stack(torch.meshgrid(yy, xx, indexing='ij'), dim=-1).reshape(-1, 2)
        indices =  torch.rand(w2c.shape[0], height_tiles * width_tiles, device=device).argsort(dim=1)
        tiles_to_render = tile_coords[indices[:, :P]].reshape(w2c.shape[0], P, 2)#.to(torch.int32)
        
        local_tile_ids = tiles_to_render[..., 0] * width_tiles + tiles_to_render[..., 1]
        i = local_tile_ids.argsort(dim=1)
        tiles_to_render = tiles_to_render.take_along_dim(i.unsqueeze(-1), dim=1)

        # mask = torch.zeros(w2c.shape[0], height_tiles, width_tiles, device=device, dtype=torch.bool)
        # camera_ids = torch.arange(w2c.shape[0], device=device, dtype=torch.int32)
        # tile_y = tiles_to_render[:, :, 0]
        # tile_x = tiles_to_render[:, :, 1]
        # C = w2c.shape[0]
        # mask[camera_ids.unsqueeze(1), tile_y, tile_x] = True
        # print(mask.sum(dim=(1, 2)))
        
        images_nhwc, alphas = gs.tile_sparse_render_images(
            tiles_to_render=tiles_to_render.contiguous(), # [C, T, 2]
            world_to_camera_matrices=w2c.contiguous(),
            projection_matrices=K.contiguous(),
            image_width=width,
            image_height=height,
            near=0.1,
            far=100.0,
            sh_degree_to_use=3,
            tile_size=16,
            min_radius_2d=min_radius_2d,
            eps_2d=0.3,
        )
        return images_nhwc, alphas
    
    def render_gsplat_packed(w2c: torch.Tensor, K: torch.Tensor):
        images_nhwc, alphas, meta = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities,
            colors=torch.cat([gs.sh0, gs.shN], dim=1),
            viewmats=w2c.contiguous(),
            Ks=K.contiguous(),
            width=width,
            height=height,
            near_plane=0.1,
            far_plane=100.0,
            sh_degree=3,
            tile_size=16,
            radius_clip=min_radius_2d,
            eps2d=0.3,
            packed=True,
        )
        return images_nhwc, alphas
    
    def render_gsplat_unpacked(w2c: torch.Tensor, K: torch.Tensor):
        images_nhwc, alphas, meta = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities,
            colors=torch.cat([gs.sh0, gs.shN], dim=1),
            viewmats=w2c.contiguous(),
            Ks=K.contiguous(),
            width=width,
            height=height,
            near_plane=0.1,
            far_plane=100.0,
            sh_degree=3,
            tile_size=16,
            radius_clip=min_radius_2d,
            eps2d=0.3,
            packed=False,
        )
        return images_nhwc, alphas
    
    @torch.no_grad()
    def benchmark_rendering(
        render_fn,
        num_cameras: int,
        w2c: torch.Tensor,
        K: torch.Tensor,
        name="render"
    ):
        w2c = w2c.expand(num_cameras, 4, 4).clone()
        Ks = K.expand(num_cameras, 3, 3).clone()
        
        nvtx.range_push(f"benchmark_{name}_setup")
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        nvtx.range_pop()
        
        nvtx.range_push(f"benchmark_{name}_warmup")
        for _ in range(num_warmup):
            nvtx.range_push(f"{name}_warmup_iter")
            render_fn(w2c, Ks)
            nvtx.range_pop()
        nvtx.range_pop()
        
        # Reset after warmup to measure only benchmark iterations
        nvtx.range_push(f"benchmark_{name}_reset")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        nvtx.range_pop()
        
        times = []
        memory_stats_list = []
        nvtx.range_push(f"benchmark_{name}_iterations")
        for i in tqdm(range(num_iterations), desc="Rendering"):
            nvtx.range_push(f"{name}_iter_{i}")
            
            # Track memory before render
            mem_before = torch.cuda.memory_stats()
            
            start = time.perf_counter()
            nvtx.range_push(f"{name}_render_call")
            render_fn(w2c, Ks)
            nvtx.range_pop()
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            # Track memory after render
            mem_after = torch.cuda.memory_stats()
            
            times.append(end - start)
            memory_stats_list.append({
                "allocated_mb": mem_after["allocated_bytes.all.current"] / (1024 ** 2),
                "reserved_mb": mem_after["reserved_bytes.all.current"] / (1024 ** 2),
                "allocated_delta_mb": (mem_after["allocated_bytes.all.current"] - mem_before["allocated_bytes.all.current"]) / (1024 ** 2),
            })
            
            nvtx.range_pop()
        nvtx.range_pop()
        
        # Get peak memory usage in bytes
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        
        # Aggregate memory stats
        avg_allocated = np.mean([s["allocated_mb"] for s in memory_stats_list])
        avg_reserved = np.mean([s["reserved_mb"] for s in memory_stats_list])
        max_allocated = np.max([s["allocated_mb"] for s in memory_stats_list])
        FPS = num_cameras / np.mean(times)
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
            "peak_memory_mb": peak_memory_mb,
            "avg_allocated_mb": avg_allocated,
            "avg_reserved_mb": avg_reserved,
            "max_allocated_mb": max_allocated,
            "FPS": FPS,
        }
    
    FUNCS = {
        "fvdb": render_fvdb,
        "fvdb_tile": render_fvdb_tile,
        "gsplat_packed": render_gsplat_packed,
        "gsplat_unpacked": render_gsplat_unpacked,
    }

    # Store results for visualization
    results = defaultdict(lambda: defaultdict(list))
    for func_name, func in FUNCS.items():
        results[func_name]['mean_ms'] = []
        results[func_name]['std_ms'] = []
        results[func_name]['peak_memory_mb'] = []
        results[func_name]['avg_allocated_mb'] = []
        results[func_name]['avg_reserved_mb'] = []
        results[func_name]['FPS'] = []
    
    for func_name, func in FUNCS.items():
        for num_cameras in NUM_CAMERAS:
            print(f"Benchmarking {func_name} rendering...")
            nvtx.range_push("benchmark_fvdb_full")
            stats = benchmark_rendering(func, num_cameras, w2c, K, name=func_name)
            nvtx.range_pop()
            print(f"  Peak GPU Memory: {stats['peak_memory_mb']:.2f} MB")
            print(f"  Avg Allocated: {stats['avg_allocated_mb']:.2f} MB")
            print(f"  Avg Reserved: {stats['avg_reserved_mb']:.2f} MB")
            print(f"  Mean Time: {stats['mean_ms']*1000:.2f} ms")
            print(f"  FPS: {stats['FPS']:.2f}")
            
            # Store results
            results[func_name]['mean_ms'].append(stats['mean_ms'] * 1000)  # Convert to ms
            results[func_name]['std_ms'].append(stats['std_ms'] * 1000)
            results[func_name]['peak_memory_mb'].append(stats['peak_memory_mb'])
            results[func_name]['avg_allocated_mb'].append(stats['avg_allocated_mb'])
            results[func_name]['avg_reserved_mb'].append(stats['avg_reserved_mb'])
            results[func_name]['FPS'].append(stats['FPS'])
    
    # Create visualizations
    num_cameras = np.array(NUM_CAMERAS)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rendering Benchmark: FVDB vs FVDB Tile-Sparse vs gsplat (Packed & Unpacked)', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean rendering time (ms)
    ax1 = axes[0, 0]
    ax1.errorbar(num_cameras, results['fvdb']['mean_ms'], 
                 yerr=results['fvdb']['std_ms'], 
                 marker='o', label='FVDB', linewidth=2, capsize=5, capthick=2)
    ax1.errorbar(num_cameras, results['fvdb_tile']['mean_ms'], 
                 yerr=results['fvdb_tile']['std_ms'], 
                 marker='s', label='FVDB Tile-Sparse', linewidth=2, capsize=5, capthick=2)
    ax1.errorbar(num_cameras, results['gsplat_packed']['mean_ms'], 
                 yerr=results['gsplat_packed']['std_ms'], 
                 marker='^', label='gsplat (Packed)', linewidth=2, capsize=5, capthick=2)
    ax1.errorbar(num_cameras, results['gsplat_unpacked']['mean_ms'], 
                 yerr=results['gsplat_unpacked']['std_ms'], 
                 marker='v', label='gsplat (Unpacked)', linewidth=2, capsize=5, capthick=2)
    ax1.set_xlabel('Number of Cameras', fontsize=12)
    ax1.set_ylabel('Mean Rendering Time (ms)', fontsize=12)
    ax1.set_title('Rendering Time vs Number of Cameras', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: FPS
    ax2 = axes[0, 1]
    ax2.plot(num_cameras, results['fvdb']['FPS'], marker='o', label='FVDB', linewidth=2)
    ax2.plot(num_cameras, results['fvdb_tile']['FPS'], marker='s', label='FVDB Tile-Sparse', linewidth=2)
    ax2.plot(num_cameras, results['gsplat_packed']['FPS'], marker='^', label='gsplat (Packed)', linewidth=2)
    ax2.plot(num_cameras, results['gsplat_unpacked']['FPS'], marker='v', label='gsplat (Unpacked)', linewidth=2)
    ax2.set_xlabel('Number of Cameras', fontsize=12)
    ax2.set_ylabel('Frames Per Second (FPS)', fontsize=12)
    ax2.set_title('FPS vs Number of Cameras', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Peak GPU Memory
    ax3 = axes[1, 0]
    ax3.plot(num_cameras, results['fvdb']['peak_memory_mb'], marker='o', label='FVDB', linewidth=2)
    ax3.plot(num_cameras, results['fvdb_tile']['peak_memory_mb'], marker='s', label='FVDB Tile-Sparse', linewidth=2)
    ax3.plot(num_cameras, results['gsplat_packed']['peak_memory_mb'], marker='^', label='gsplat (Packed)', linewidth=2)
    ax3.plot(num_cameras, results['gsplat_unpacked']['peak_memory_mb'], marker='v', label='gsplat (Unpacked)', linewidth=2)
    ax3.set_xlabel('Number of Cameras', fontsize=12)
    ax3.set_ylabel('Peak GPU Memory (MB)', fontsize=12)
    ax3.set_title('Peak GPU Memory Usage', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average Allocated Memory
    ax4 = axes[1, 1]
    ax4.plot(num_cameras, results['fvdb']['avg_allocated_mb'], marker='o', label='FVDB', linewidth=2)
    ax4.plot(num_cameras, results['fvdb_tile']['avg_allocated_mb'], marker='s', label='FVDB Tile-Sparse', linewidth=2)
    ax4.plot(num_cameras, results['gsplat_packed']['avg_allocated_mb'], marker='^', label='gsplat (Packed)', linewidth=2)
    ax4.plot(num_cameras, results['gsplat_unpacked']['avg_allocated_mb'], marker='v', label='gsplat (Unpacked)', linewidth=2)
    ax4.set_xlabel('Number of Cameras', fontsize=12)
    ax4.set_ylabel('Average Allocated Memory (MB)', fontsize=12)
    ax4.set_title('Average Allocated GPU Memory', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'benchmark_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also show the plot
    plt.show()


