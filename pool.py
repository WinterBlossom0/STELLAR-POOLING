import numpy as np
from PIL import Image

def stellar_pool(input_tensor, channels, kernel_size):
    height, width = input_tensor.shape[:2]
    
    # Handle grayscale by converting to 3-channel
    is_grayscale = False
    if len(input_tensor.shape) == 2:
        is_grayscale = True
        # Convert grayscale to 3 identical channels
        input_tensor = np.stack([input_tensor, input_tensor, input_tensor], axis=2)
        channels = 3  # Process as RGB
    elif len(input_tensor.shape) == 3 and input_tensor.shape[2] == 1:
        is_grayscale = True
        # Expand single channel to 3 identical channels
        input_tensor = np.concatenate([input_tensor, input_tensor, input_tensor], axis=2)
        channels = 3
    elif len(input_tensor.shape) == 3:
        # Always use only RGB channels (ignore alpha)
        actual_channels = min(input_tensor.shape[2], 3)  # Max 3 channels
        input_tensor = input_tensor[:, :, :3]  # Only take RGB
        channels = 3
    
    # Handle any kernel size greater than 1
    if kernel_size < 2:
        raise ValueError(f"Kernel size {kernel_size} not supported. Only kernel sizes >= 2 are supported.")
    
    # Apply pooling and convert back to grayscale if needed
    result = apply_pooling(input_tensor, channels, kernel_size)
    
    # Convert back to grayscale if input was grayscale
    if is_grayscale:
        result = result[:, :, 0]
    
    return result

def apply_pooling(input_tensor, channels, kernel_size):
    """Apply stellar pooling algorithm with any kernel size"""
    height, width = input_tensor.shape[:2]
    
    # Calculate exact output dimensions
    output_height = height // kernel_size
    output_width = width // kernel_size
    
    # Always process as RGB (channels=3)
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
            processed_pixel = process_block(block, 3, input_tensor, i, j, kernel_size, printed, global_entropies)
            output[out_i, out_j, :] = processed_pixel
            printed = True
    
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
    # Only handle RGB case since grayscale is pre-converted to RGB
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
    kernel_pixels = kernel_size * kernel_size
    votes_per_channel = int((kernel_pixels * 3) / 12)
    
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
            
            # Use combined_values BUT exclude extremes for median calculation
            sorted_channel = np.sort(channel_values)
            # Exclude min/max when calculating median from combined values
            median_val = np.median(sorted_channel[1:-1])
            
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
            
            # CORRECT: Exclude min/max from KERNEL BLOCK for median calculation
            # GENERALIZED: Use combined_values for median calculation
            sorted_channel = np.sort(channel_values)

            # Exclude votes_per_channel min AND votes_per_channel max values from combined value
            median_val = np.median(sorted_channel[votes_per_channel:-votes_per_channel])

            
            # Calculate local entropy from ALL combined values (no exclusions)
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
    # selected_pixels should ALWAYS have content
    result_pixel = np.mean(selected_pixels, axis=0)
    return result_pixel.astype(block.dtype)

# Main execution
if __name__ == "__main__":
    # Load image
    image = Image.open("image.png")
    image_tensor = np.array(image)[:, :, :3]  # Only take RGB, ignore alpha
    
    # Apply stellar pooling with different kernel sizes
    kernel_size = 6  # Now supports any kernel size >= 2
    
    result = stellar_pool(image_tensor, 3, kernel_size)
    
    # Convert result back to image and save
    result_image = Image.fromarray(result.astype(np.uint8))
    result_image.save(f"custom_{kernel_size}x{kernel_size}.png")
    
    compression_ratio = (image_tensor.shape[0] * image_tensor.shape[1]) / (result.shape[0] * result.shape[1])
    print(f"Compression ratio: {compression_ratio:.1f}x smaller")