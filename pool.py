import numpy as np
from PIL import Image

def stellar_pool(input_tensor, channels, kernel_size):
    height, width = input_tensor.shape[:2]
    
    # Always use only RGB channels (ignore alpha)
    if len(input_tensor.shape) == 3:
        actual_channels = min(input_tensor.shape[2], 3)  # Max 3 channels
        input_tensor = input_tensor[:, :, :3]  # Only take RGB
        channels = 3
    
    # Handle different kernel sizes
    if kernel_size == 2:
        # Apply 2x2 pooling once
        return apply_2x2_pooling(input_tensor, channels)
    elif kernel_size == 4:
        # Apply 2x2 pooling twice
        # print("Applying 2x2 pooling twice for 4x4 kernel")
        
        # First pass: 2x2 pooling
        intermediate_result = apply_2x2_pooling(input_tensor, channels)
        # print(f"After first 2x2 pass: {input_tensor.shape} -> {intermediate_result.shape}")
        
        # Second pass: 2x2 pooling on the result
        final_result = apply_2x2_pooling(intermediate_result, channels)
        # print(f"After second 2x2 pass: {intermediate_result.shape} -> {final_result.shape}")
        
        return final_result
    elif kernel_size == 6:
        # Apply 2x2 pooling three times
        # print("Applying 2x2 pooling three times for 6x6 kernel")
        
        # First pass
        result1 = apply_2x2_pooling(input_tensor, channels)
        # print(f"After first 2x2 pass: {input_tensor.shape} -> {result1.shape}")
        
        # Second pass
        result2 = apply_2x2_pooling(result1, channels)
        # print(f"After second 2x2 pass: {result1.shape} -> {result2.shape}")
        
        # Third pass
        final_result = apply_2x2_pooling(result2, channels)
        # print(f"After third 2x2 pass: {result2.shape} -> {final_result.shape}")
        
        return final_result
    else:
        raise ValueError(f"Kernel size {kernel_size} not supported. Only even numbers (2, 4, 6, 8...) are supported.")

def apply_2x2_pooling(input_tensor, channels):
    """Apply 2x2 stellar pooling algorithm"""
    height, width = input_tensor.shape[:2]
    kernel_size = 2
    
    if channels == 3:
        red_entropy = calculate_entropy(input_tensor[:, :, 0])
        green_entropy = calculate_entropy(input_tensor[:, :, 1])
        blue_entropy = calculate_entropy(input_tensor[:, :, 2])
        global_entropies = [red_entropy, green_entropy, blue_entropy]
        
        output = np.zeros((height // kernel_size, width // kernel_size, 3), dtype=input_tensor.dtype)
        printed = False
        
        for i in range(0, height - kernel_size + 1, kernel_size):
            for j in range(0, width - kernel_size + 1, kernel_size):
                block = input_tensor[i:i+kernel_size, j:j+kernel_size, :]
                processed_pixel = process_block(block, channels, input_tensor, i, j, kernel_size, printed, global_entropies)
                output[i//kernel_size, j//kernel_size, :] = processed_pixel
                printed = True
                
    else:
        output = np.zeros((height // kernel_size, width // kernel_size), dtype=input_tensor.dtype)
        printed = False
        
        for i in range(0, height - kernel_size + 1, kernel_size):
            for j in range(0, width - kernel_size + 1, kernel_size):
                block = input_tensor[i:i+kernel_size, j:j+kernel_size]
                processed_pixel = process_block(block, channels, input_tensor, i, j, kernel_size, printed, None)
                output[i//kernel_size, j//kernel_size] = processed_pixel
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
    if channels == 3:
        max_vals = np.max(block, axis=(0,1))
        min_vals = np.min(block, axis=(0,1))
        
        max_positions = []
        min_positions = []
        for c in range(3):  # Only RGB channels
            max_pos = np.unravel_index(np.argmax(block[:,:,c]), block[:,:,c].shape)
            min_pos = np.unravel_index(np.argmin(block[:,:,c]), block[:,:,c].shape)
            max_positions.append(max_pos)
            min_positions.append(min_pos)
        
        # if not printed:
        #     print("Max values:", max_vals)
        #     print("Min values:", min_vals)
        #     print("Max positions:", max_positions)
        #     print("Min positions:", min_positions)
        
        height, width = full_image.shape[:2]
        
        start_i = max(0, i - 1)
        end_i = min(height, i + kernel_size + 1)
        start_j = max(0, j - 1)
        end_j = min(width, j + kernel_size + 1)
        
        neighbor_block = full_image[start_i:end_i, start_j:end_j, :]
        
        mask = np.ones(neighbor_block.shape[:2], dtype=bool)
        rel_i = max(0, 1 - (i - start_i))
        rel_j = max(0, 1 - (j - start_j))
        mask[rel_i:rel_i+kernel_size, rel_j:rel_j+kernel_size] = False
        
        neighbor_values = neighbor_block[mask]
        
        block_reshaped = block.reshape(-1, 3)  # Always 3 channels
        combined_values = np.concatenate([block_reshaped, neighbor_values])
        
        median_values = []
        local_entropies = []
        votes = []
        
        for c in range(3):  # Only RGB channels
            # For median: exclude max and min
            channel_values_median = combined_values[:, c].copy()
            channel_values_median = np.sort(channel_values_median)
            
            min_idx = np.where(channel_values_median == min_vals[c])[0]
            max_idx = np.where(channel_values_median == max_vals[c])[0]
            
            if len(min_idx) > 0:
                channel_values_median = np.delete(channel_values_median, min_idx[0])
            if len(max_idx) > 0:
                max_idx = np.where(channel_values_median == max_vals[c])[0]
                if len(max_idx) > 0:
                    channel_values_median = np.delete(channel_values_median, max_idx[0])
            
            median_val = np.median(channel_values_median)
            median_values.append(median_val)
            
            # For entropy: include max and min
            channel_values_entropy = combined_values[:, c]
            local_entropy = calculate_entropy(channel_values_entropy)
            local_entropies.append(local_entropy)
            
            # Voting logic
            global_entropy = global_entropies[c]
            
            # Calculate distances from median for max and min pixels in the 2x2 block
            max_distance = abs(max_vals[c] - median_val)
            min_distance = abs(min_vals[c] - median_val)
            
            if local_entropy < global_entropy:
                # Vote for pixel with GREATER distance to median
                if max_distance > min_distance:
                    vote = "max"  # Max pixel is farther from median
                else:
                    vote = "min"  # Min pixel is farther from median
            else:
                # Vote for pixel with SMALLER distance to median
                if max_distance < min_distance:
                    vote = "max"  # Max pixel is closer to median
                else:
                    vote = "min"  # Min pixel is closer to median
            
            votes.append(vote)
        
        # if not printed:
        #     print("Median values:", median_values)
        #     print("Local entropies:", local_entropies)
        #     print("Global entropies:", global_entropies)
        #     print("Votes (R,G,B):", votes)
        
        # Count votes and select pixels
        selected_pixels = []
        for c in range(3):
            if votes[c] == "max":
                selected_pixels.append(block[max_positions[c][0], max_positions[c][1], :].astype(np.float64))
            else:
                selected_pixels.append(block[min_positions[c][0], min_positions[c][1], :].astype(np.float64))

        # Add selected pixels and divide by 3
        result_pixel = (selected_pixels[0] + selected_pixels[1] + selected_pixels[2]) / 3.0
        
        # if not printed:
        #     print("Selected pixels:", selected_pixels)
        #     print("Result pixel:", result_pixel)
        
        return result_pixel.astype(block.dtype)
    else:
        max_val = np.max(block)
        min_val = np.min(block)
        
        max_pos = np.unravel_index(np.argmax(block), block.shape)
        min_pos = np.unravel_index(np.argmin(block), block.shape)
        
        # if not printed:
        #     print("Max value:", max_val)
        #     print("Min value:", min_val)
        #     print("Max position:", max_pos)
        #     print("Min position:", min_pos)
        
        height, width = full_image.shape[:2]
        
        start_i = max(0, i - 1)
        end_i = min(height, i + kernel_size + 1)
        start_j = max(0, j - 1)
        end_j = min(width, j + kernel_size + 1)
        
        neighbor_block = full_image[start_i:end_i, start_j:end_j]
        
        mask = np.ones(neighbor_block.shape, dtype=bool)
        rel_i = max(0, 1 - (i - start_i))
        rel_j = max(0, 1 - (j - start_j))
        mask[rel_i:rel_i+kernel_size, rel_j:rel_j+kernel_size] = False
        
        neighbor_values = neighbor_block[mask]
        
        combined_values = np.concatenate([block.flatten(), neighbor_values])
        
        # For median: exclude max and min
        combined_values_median = np.sort(combined_values.copy())
        min_idx = np.where(combined_values_median == min_val)[0]
        max_idx = np.where(combined_values_median == max_val)[0]
        
        if len(min_idx) > 0:
            combined_values_median = np.delete(combined_values_median, min_idx[0])
        if len(max_idx) > 0:
            max_idx = np.where(combined_values_median == max_val)[0]
            if len(max_idx) > 0:
                combined_values_median = np.delete(combined_values_median, max_idx[0])
        
        median_val = np.median(combined_values_median)
        
        # For entropy: include max and min
        local_entropy = calculate_entropy(combined_values)
        
        # if not printed:
        #     print("Median value:", median_val)
        #     print("Local entropy:", local_entropy)
        
        return median_val

# Main execution
if __name__ == "__main__":
    # Load image
    image = Image.open("image.png")
    image_tensor = np.array(image)[:, :, :3]  # Only take RGB, ignore alpha
    
    print(f"Original image shape: {image_tensor.shape}")
    
    # Apply stellar pooling with different kernel sizes
    kernel_size = 4  # Change this to test different kernel sizes (2, 4, 6, 8...)
    
    result = stellar_pool(image_tensor, 3, kernel_size)
    print(f"Compressed result shape: {result.shape}")
    
    # Convert result back to image and save
    result_image = Image.fromarray(result.astype(np.uint8))
    result_image.save(f"custom_{kernel_size}x{kernel_size}.png")
    
    print(f"✓ Compressed image saved as 'custom_{kernel_size}x{kernel_size}.png'")
    compression_ratio = (image_tensor.shape[0] * image_tensor.shape[1]) / (result.shape[0] * result.shape[1])
    print(f"Compression ratio: {compression_ratio:.1f}x smaller")