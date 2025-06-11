import numpy as np
from PIL import Image

def stellar_pool(input_tensor, channels, kernel_size):
    height, width = input_tensor.shape[:2]
    
    # Always use only RGB channels (ignore alpha)
    if len(input_tensor.shape) == 3:
        actual_channels = min(input_tensor.shape[2], 3)  # Max 3 channels
        input_tensor = input_tensor[:, :, :3]  # Only take RGB
        channels = 3
    
    # Handle any kernel size greater than 1
    if kernel_size < 2:
        raise ValueError(f"Kernel size {kernel_size} not supported. Only kernel sizes >= 2 are supported.")
    
    return apply_pooling(input_tensor, channels, kernel_size)

def apply_pooling(input_tensor, channels, kernel_size):
    """Apply stellar pooling algorithm with any kernel size"""
    height, width = input_tensor.shape[:2]
    
    # Calculate exact output dimensions
    output_height = height // kernel_size
    output_width = width // kernel_size
    
    # DEBUG: Print dimensions
    print(f"Input dimensions: {height}x{width}")
    print(f"Kernel size: {kernel_size}")
    print(f"Output dimensions: {output_height}x{output_width}")
    print(f"Total output pixels: {output_height * output_width}")
    
    if channels == 3:
        red_entropy = calculate_entropy(input_tensor[:, :, 0])
        green_entropy = calculate_entropy(input_tensor[:, :, 1])
        blue_entropy = calculate_entropy(input_tensor[:, :, 2])
        global_entropies = [red_entropy, green_entropy, blue_entropy]
        
        output = np.zeros((output_height, output_width, 3), dtype=input_tensor.dtype)
        printed = False
        
        # Use output dimensions for loop range
        for out_i in range(output_height):
            for out_j in range(output_width):
                i = out_i * kernel_size
                j = out_j * kernel_size
                block = input_tensor[i:i+kernel_size, j:j+kernel_size, :]
                processed_pixel = process_block(block, channels, input_tensor, i, j, kernel_size, printed, global_entropies)
                output[out_i, out_j, :] = processed_pixel
                printed = True
    
    print(f"Final output shape: {output.shape}")
    return output

def calculate_entropy(channel):
    histogram, _ = np.histogram(channel, bins=256, range=(0, 255))
    
    # Calculate probabilities INCLUDING zero bins
    total_pixels = np.sum(histogram)
    probabilities = histogram / total_pixels
    
    # Remove only zero probabilities before log calculation (to avoid log(0))
    non_zero_probs = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    return entropy

def process_block(block, channels, full_image, i, j, kernel_size, printed, global_entropies):
    if channels == 3:
        height, width = full_image.shape[:2]
        
        # Calculate neighbor window size: kernel_size + kernel_size - 1
        neighbor_window_size = kernel_size + kernel_size - 1
        neighbor_radius = neighbor_window_size // 2
        
        start_i = max(0, i - neighbor_radius)
        end_i = min(height, i + kernel_size + neighbor_radius)
        start_j = max(0, j - neighbor_radius)
        end_j = min(width, j + kernel_size + neighbor_radius)
        
        neighbor_block = full_image[start_i:end_i, start_j:end_j, :]
        
        # Create mask to exclude the kernel area
        mask = np.ones(neighbor_block.shape[:2], dtype=bool)
        rel_i = i - start_i
        rel_j = j - start_j
        mask[rel_i:rel_i+kernel_size, rel_j:rel_j+kernel_size] = False
        
        neighbor_values = neighbor_block[mask]
        
        block_reshaped = block.reshape(-1, 3)
        combined_values = np.concatenate([block_reshaped, neighbor_values])
        
        # Calculate number of votes per channel based on 3:1 ratio from KERNEL pixels only
        kernel_pixels = kernel_size * kernel_size  # 6x6 = 36 pixels
        votes_per_channel = int((kernel_pixels * 3) / 12)  # Round down to lowest integer
        
        selected_pixels = []
        
        for c in range(3):  # RGB channels
            channel_values = combined_values[:, c].copy()
            
            if votes_per_channel == 1:
                # For 2x2 kernel: Find single min and max from KERNEL BLOCK ONLY
                block_channel = block[:, :, c].flatten()
                min_val = np.min(block_channel)
                max_val = np.max(block_channel)
                
                # Find positions in block
                min_pos = np.where(block_channel == min_val)[0]
                max_pos = np.where(block_channel == max_val)[0]
                
                if len(min_pos) > 0:
                    min_pos_2d = np.unravel_index(min_pos[0], block[:,:,c].shape)
                else:
                    min_pos_2d = (0, 0)
                    
                if len(max_pos) > 0:
                    max_pos_2d = np.unravel_index(max_pos[0], block[:,:,c].shape)
                else:
                    max_pos_2d = (0, 0)
                
                # Use combined_values for median calculation (kernel + neighbors)
                median_val = np.median(channel_values)
                
                # Calculate local entropy from combined values
                local_entropy = calculate_entropy(channel_values)
                global_entropy = global_entropies[c]
                
                # Single vote: max vs min from kernel block
                max_distance = abs(max_val - median_val)
                min_distance = abs(min_val - median_val)
                
                if local_entropy < global_entropy:
                    if max_distance > min_distance:
                        pixel_pos = max_pos_2d
                    else:
                        pixel_pos = min_pos_2d
                else:
                    if max_distance < min_distance:
                        pixel_pos = max_pos_2d
                    else:
                        pixel_pos = min_pos_2d
                
                selected_pixels.append(block[pixel_pos[0], pixel_pos[1], :].astype(np.float64))
            
            else:
                # For 3x3+ kernels: Find multiple min/max from KERNEL BLOCK ONLY
                block_channel = block[:, :, c].flatten()
                sorted_indices = np.argsort(block_channel)
                sorted_block_values = block_channel[sorted_indices]
                
                min_values = []
                max_values = []
                min_positions = []
                max_positions = []
                
                # Get the votes_per_channel smallest and largest values from KERNEL BLOCK
                for v in range(votes_per_channel):
                    min_val = sorted_block_values[v]  # v-th smallest from kernel
                    max_val = sorted_block_values[-(v+1)]  # v-th largest from kernel
                    
                    min_values.append(min_val)
                    max_values.append(max_val)
                    
                    # Find positions in block
                    min_pos = np.where(block_channel == min_val)[0]
                    max_pos = np.where(block_channel == max_val)[0]
                    
                    if len(min_pos) > 0:
                        min_pos_2d = np.unravel_index(min_pos[0], block[:,:,c].shape)
                        min_positions.append(min_pos_2d)
                    else:
                        min_positions.append((0, 0))
                        
                    if len(max_pos) > 0:
                        max_pos_2d = np.unravel_index(max_pos[0], block[:,:,c].shape)
                        max_positions.append(max_pos_2d)
                    else:
                        max_positions.append((0, 0))
                
                # Use combined_values for median calculation (excluding extremes)
                median_values = channel_values  # Use all combined values for median
                median_val = np.median(median_values)
                
                # Calculate local entropy from combined values
                local_entropy = calculate_entropy(channel_values)
                global_entropy = global_entropies[c]
                
                # Vote for each pair
                for v in range(votes_per_channel):
                    max_distance = abs(max_values[v] - median_val)
                    min_distance = abs(min_values[v] - median_val)
                    
                    if local_entropy < global_entropy:
                        if max_distance > min_distance:
                            pixel_pos = max_positions[v]
                        else:
                            pixel_pos = min_positions[v]
                    else:
                        if max_distance < min_distance:
                            pixel_pos = max_positions[v]
                        else:
                            pixel_pos = min_positions[v]
                    
                    selected_pixels.append(block[pixel_pos[0], pixel_pos[1], :].astype(np.float64))
                
                if votes_per_channel == 0:
                    center_pos = (kernel_size//2, kernel_size//2)
                    selected_pixels.append(block[center_pos[0], center_pos[1], :].astype(np.float64))
        
        # Use SIMPLE averaging like pool_og.py (no duplicate handling)
        if len(selected_pixels) > 0:
            result_pixel = np.sum(selected_pixels, axis=0) / len(selected_pixels)
        else:
            # Fallback to center pixel
            center_pos = (kernel_size//2, kernel_size//2)
            result_pixel = block[center_pos[0], center_pos[1], :].astype(np.float64)
        
        return result_pixel.astype(block.dtype)
    
    else:
        # Convert grayscale to 3 channels (RGB with same values) to follow same logic
        if len(full_image.shape) == 2:
            # Expand grayscale to 3 channels
            full_image_3ch = np.stack([full_image, full_image, full_image], axis=2)
            block_3ch = np.stack([block, block, block], axis=2)
        else:
            full_image_3ch = full_image
            block_3ch = block
        
        # Now process as RGB with 3 identical channels
        height, width = full_image_3ch.shape[:2]
        
        neighbor_window_size = kernel_size + kernel_size - 1
        neighbor_radius = neighbor_window_size // 2
        
        start_i = max(0, i - neighbor_radius)
        end_i = min(height, i + kernel_size + neighbor_radius)
        start_j = max(0, j - neighbor_radius)
        end_j = min(width, j + kernel_size + neighbor_radius)
        
        neighbor_block = full_image_3ch[start_i:end_i, start_j:end_j, :]
        
        mask = np.ones(neighbor_block.shape[:2], dtype=bool)
        rel_i = i - start_i
        rel_j = j - start_j
        mask[rel_i:rel_i+kernel_size, rel_j:rel_j+kernel_size] = False
        
        neighbor_values = neighbor_block[mask]
        
        block_reshaped = block_3ch.reshape(-1, 3)
        combined_values = np.concatenate([block_reshaped, neighbor_values])
        
        # Calculate global entropy from the single grayscale channel (all 3 are identical)
        global_entropy = calculate_entropy(full_image_3ch[:, :, 0])
        global_entropies = [global_entropy, global_entropy, global_entropy]
        
        # Use same RGB processing logic
        kernel_pixels = kernel_size * kernel_size
        votes_per_channel = int((kernel_pixels * 3) / 12)
        
        selected_pixels = []
        
        for c in range(3):  # Process all 3 identical channels
            channel_values = combined_values[:, c].copy()
            sorted_indices = np.argsort(channel_values)
            sorted_values = channel_values[sorted_indices]
            
            min_values = []
            max_values = []
            min_positions = []
            max_positions = []
            
            for v in range(votes_per_channel):
                min_val = sorted_values[v]
                max_val = sorted_values[-(v+1)]
                
                min_values.append(min_val)
                max_values.append(max_val)
                
                block_flat = block_3ch[:,:,c].flatten()
                min_pos = np.where(block_flat == min_val)[0]
                max_pos = np.where(block_flat == max_val)[0]
                
                if len(min_pos) > 0:
                    min_pos_2d = np.unravel_index(min_pos[0], block_3ch[:,:,c].shape)
                    min_positions.append(min_pos_2d)
                else:
                    min_positions.append((0, 0))
                    
                if len(max_pos) > 0:
                    max_pos_2d = np.unravel_index(max_pos[0], block_3ch[:,:,c].shape)
                    max_positions.append(max_pos_2d)
                else:
                    max_positions.append((0, 0))
            
            median_values = sorted_values[votes_per_channel:-votes_per_channel] if votes_per_channel > 0 else sorted_values
            median_val = np.median(median_values)
            
            local_entropy = calculate_entropy(channel_values)
            global_entropy = global_entropies[c]
            
            channel_selected_pixels = []
            for v in range(votes_per_channel):
                max_distance = abs(max_values[v] - median_val)
                min_distance = abs(min_values[v] - median_val)
                
                if local_entropy < global_entropy:
                    if max_distance > min_distance:
                        pixel_pos = max_positions[v]
                    else:
                        pixel_pos = min_positions[v]
                else:
                    if max_distance < min_distance:
                        pixel_pos = max_positions[v]
                    else:
                        pixel_pos = min_positions[v]
                
                channel_selected_pixels.append(block_3ch[pixel_pos[0], pixel_pos[1], :].astype(np.float64))
            
            if len(channel_selected_pixels) == 0:
                center_pos = (kernel_size//2, kernel_size//2)
                channel_selected_pixels.append(block_3ch[center_pos[0], center_pos[1], :].astype(np.float64))
            
            selected_pixels.extend(channel_selected_pixels)
        
        # Handle duplicates and calculate weighted average
        unique_pixels = {}
        for pixel in selected_pixels:
            pixel_key = tuple(pixel)
            if pixel_key in unique_pixels:
                unique_pixels[pixel_key] += 1
            else:
                unique_pixels[pixel_key] = 1
        
        total_weight = sum(unique_pixels.values())
        result_pixel = np.zeros(3, dtype=np.float64)
        
        for pixel_tuple, weight in unique_pixels.items():
            pixel_array = np.array(pixel_tuple)
            result_pixel += pixel_array * weight
        
        result_pixel /= total_weight
        
        # Return only the first channel value (since all 3 are identical for grayscale)
        return result_pixel[0].astype(block.dtype)

# Main execution
if __name__ == "__main__":
    # Load image
    image = Image.open("image.png")
    image_tensor = np.array(image)[:, :, :3]  # Only take RGB, ignore alpha
    
    # Apply stellar pooling with different kernel sizes
    kernel_size = 2  # Now supports any kernel size >= 2
    
    result = stellar_pool(image_tensor, 3, kernel_size)
    
    # Convert result back to image and save
    result_image = Image.fromarray(result.astype(np.uint8))
    result_image.save(f"custom_{kernel_size}x{kernel_size}.png")
    
    compression_ratio = (image_tensor.shape[0] * image_tensor.shape[1]) / (result.shape[0] * result.shape[1])
    print(f"Compression ratio: {compression_ratio:.1f}x smaller")