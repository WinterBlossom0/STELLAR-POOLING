import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as F

def get_remaining_pixels(flat_channel):
    """Remove only ONE instance of min and ONE instance of max, keep everything else"""
    channel_min = np.min(flat_channel)
    channel_max = np.max(flat_channel)
    
    values_list = list(flat_channel)
    
    # Remove only ONE instance of min
    if channel_min in values_list:
        values_list.remove(channel_min)
    
    # Remove only ONE instance of max (if different from min)
    if channel_max != channel_min and channel_max in values_list:
        values_list.remove(channel_max)
    
    return values_list

def get_neighboring_pixels_correct(img_array, block_i, block_j, block_size, channel):
    """Get neighbors for 4x4 block elements, excluding block positions"""
    height, width = img_array.shape[:2]
    neighbor_positions = set()
    
    # For each element in the 4x4 block
    for i in range(block_i, min(block_i + block_size, height)):
        for j in range(block_j, min(block_j + block_size, width)):
            # Get 8 neighbors around this element
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # Skip the element itself
                    
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        # Check if this neighbor is NOT part of our 4x4 block
                        if not (block_i <= ni < block_i + block_size and 
                                block_j <= nj < block_j + block_size):
                            neighbor_positions.add((ni, nj))
    
    # Extract values from unique neighbor positions
    neighbors = []
    for ni, nj in neighbor_positions:
        if len(img_array.shape) == 3:
            pixel_val = img_array[ni, nj, channel]
        else:
            pixel_val = img_array[ni, nj]
        neighbors.append(pixel_val)
    
    return neighbors

def compress_block_custom(block, img_array, block_i, block_j, block_size):
    """Algorithm: 4x4 blocks with median-based selection and ratio blending"""
    if len(block.shape) == 3:
        height, width, channels = block.shape
        
        original_pixels = block.reshape(-1, channels)
        channel_selections = []
        
        for c in range(channels):
            channel_block = block[:, :, c]
            flat_channel = channel_block.flatten()
            
            # Get remaining pixels after removing ONE min and ONE max
            remaining = get_remaining_pixels(flat_channel)
            
            # Get neighboring pixels (all neighbors including diagonals)
            neighboring_pixels = get_neighboring_pixels_correct(img_array, block_i, block_j, block_size, c)
            
            # Calculate LOCAL MEDIAN from remaining + neighbors
            combined_pixels = remaining + neighboring_pixels
            local_median = np.median(combined_pixels)
            
            # Find which pixel this channel wants to select
            best_pixel_idx = 0
            best_distance = 0
            
            # Update the pixel selection logic to use median
            for pixel_idx, pixel in enumerate(original_pixels):
                pixel_channel_value = pixel[c]
                distance = abs(float(pixel_channel_value) - float(local_median))
                
                if distance > best_distance:
                    best_distance = distance
                    best_pixel_idx = pixel_idx
            
            channel_selections.append(best_pixel_idx)
        
        # Count votes and blend
        from collections import Counter
        vote_counts = Counter(channel_selections)
        
        total_votes = len(channel_selections)
        blend_ratios = {}
        for pixel_idx, votes in vote_counts.items():
            blend_ratios[pixel_idx] = votes / total_votes
        
        # Create blended pixel
        blended_pixel = np.zeros(channels, dtype=float)
        
        for pixel_idx, ratio in blend_ratios.items():
            selected_pixel = original_pixels[pixel_idx]
            for c in range(channels):
                blended_pixel[c] += ratio * selected_pixel[c]
        
        # Convert to uint8
        blended_pixel = np.clip(blended_pixel, 0, 255).astype(np.uint8)
        
        # Fill entire block with blended pixel
        result_block = np.zeros_like(block)
        result_block[:, :] = blended_pixel
        
        return result_block
    
    else:
        # Grayscale version with 4x4 blocks
        flat_block = block.flatten()
        
        remaining = get_remaining_pixels(flat_block)
        neighboring_pixels = get_neighboring_pixels_correct(img_array, block_i, block_j, block_size, 0)
        
        combined_pixels = remaining + neighboring_pixels
        local_median = np.median(combined_pixels)
        
        min_val = np.min(flat_block)
        max_val = np.max(flat_block)
        
        min_distance = abs(float(min_val) - float(local_median))
        max_distance = abs(float(max_val) - float(local_median))
        
        if max_distance > min_distance:
            selected_value = max_val
        else:
            selected_value = min_val
        
        return np.full_like(block, selected_value)

def apply_pytorch_pooling(img_array, pooling_type):
    """Apply PyTorch pooling operations with 4x4 blocks"""
    # Convert numpy array to PyTorch tensor
    if len(img_array.shape) == 3:
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
    
    # Apply pooling with kernel_size=4, stride=4 (4x4 blocks)
    if pooling_type == "max":
        pooled = F.max_pool2d(tensor, kernel_size=4, stride=4)
    elif pooling_type == "min":
        pooled = -F.max_pool2d(-tensor, kernel_size=4, stride=4)
    elif pooling_type == "average":
        pooled = F.avg_pool2d(tensor, kernel_size=4, stride=4)
    
    # Convert back to numpy and original shape
    if len(img_array.shape) == 3:
        result = pooled.squeeze(0).permute(1, 2, 0).numpy()
    else:
        result = pooled.squeeze(0).squeeze(0).numpy()
    
    # Upsample back to original size by repeating each pooled value in 4x4 blocks
    if len(img_array.shape) == 3:
        height, width, channels = img_array.shape
        upsampled = np.zeros_like(img_array)
        
        for i in range(0, height, 4):
            for j in range(0, width, 4):
                end_i = min(i + 4, height)
                end_j = min(j + 4, width)
                
                pool_i, pool_j = i // 4, j // 4
                if pool_i < result.shape[0] and pool_j < result.shape[1]:
                    pooled_value = result[pool_i, pool_j]
                    upsampled[i:end_i, j:end_j] = pooled_value
    else:
        height, width = img_array.shape
        upsampled = np.zeros_like(img_array)
        
        for i in range(0, height, 4):
            for j in range(0, width, 4):
                end_i = min(i + 4, height)
                end_j = min(j + 4, width)
                
                pool_i, pool_j = i // 4, j // 4
                if pool_i < result.shape[0] and pool_j < result.shape[1]:
                    pooled_value = result[pool_i, pool_j]
                    upsampled[i:end_i, j:end_j] = pooled_value
    
    return upsampled.astype(np.uint8)

def compress_image(image_path, method, output_path):
    """Compress image using specified compression technique with 4x4 blocks"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:
            height, width, channels = img_array.shape
            print(f"Processing color image: {image_path}")
            print(f"Image dimensions: {width}x{height}x{channels}")
        else:
            height, width = img_array.shape
            channels = 1
            print(f"Processing grayscale image: {image_path}")
            print(f"Image dimensions: {width}x{height}")
        
        print(f"Method: {method.upper()} - Block size: 4x4")
        
        if method == "custom":
            print("Using 4x4 blocks with median-based selection and diagonal neighbors!")
            
            compressed_array = np.zeros_like(img_array)
            block_size = 4  # 4x4 blocks
            blocks_processed = 0
            
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    end_i = min(i + block_size, height)
                    end_j = min(j + block_size, width)
                    
                    block = img_array[i:end_i, j:end_j]
                    compressed_block = compress_block_custom(block, img_array, i, j, block_size)
                    compressed_array[i:end_i, j:end_j] = compressed_block
                    blocks_processed += 1
            
            print(f"Processed {blocks_processed} 4x4 blocks")
            
        else:
            compressed_array = apply_pytorch_pooling(img_array, method)
            print(f"Applied PyTorch {method} pooling with 4x4 blocks")
        
        compressed_img = Image.fromarray(compressed_array.astype(np.uint8))
        compressed_img.save(output_path)
        
        if os.path.exists(output_path):
            original_size = os.path.getsize(image_path)
            compressed_size = os.path.getsize(output_path)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            print(f"✓ Compressed image saved: {output_path}")
            print(f"Original file size: {original_size:,} bytes")
            print(f"Compressed file size: {compressed_size:,} bytes")
            print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return output_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    image_path = "image.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found in workspace!")
        return
    
    print("Running ALL Compression Algorithms (4x4 blocks)")
    print("=" * 80)
    
    methods = [
        ("custom", "custom.png"),
        ("max", "max.png"),
        ("min", "min.png"),
        ("average", "average.png")
    ]
    
    for method, output_file in methods:
        print(f"\n🔄 Running {method.upper()} compression...")
        print("-" * 40)
        
        result = compress_image(image_path, method, output_file)
        
        if result:
            print(f"✅ {method.upper()} compression completed!")
        else:
            print(f"❌ {method.upper()} compression failed!")
    
    print(f"\n{'=' * 80}")
    print("ALL COMPRESSIONS COMPLETED!")
    print("Generated files: custom.png, max.png, min.png, average.png")

if __name__ == "__main__":
    main()