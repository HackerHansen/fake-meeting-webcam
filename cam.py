import cv2
import numpy as np
import random
import pyvirtualcam
import time

def process_frame(frame, scale_factor, noise_level, compression_quality, original_dimensions):
    # Resize to lower resolution
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Add random noise
    noise = np.random.randint(0, 100, small_frame.shape, dtype=np.uint8)
    noisy_frame = cv2.addWeighted(small_frame, 1 - noise_level, noise, noise_level, 0)
    
    # Compress the image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
    _, buffer = cv2.imencode('.jpg', noisy_frame, encode_param)
    compressed_frame = cv2.imdecode(buffer, 1)
    
    # Resize back to original resolution
    upscaled_frame = cv2.resize(compressed_frame, original_dimensions)
    
    return upscaled_frame

def run_cam(video_path, scale_factor=0.07, noise_level=0.2, compression_quality=30, drop_rate=0.1, max_delay=0.35):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("The script had a stroke trying to open the video file")
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_dimensions = (original_width, original_height)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    with pyvirtualcam.Camera(width=original_width, height=original_height, fps=fps) as cam:
        print(f'Using virtual camera: {cam.device}')
        
        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Frame", 192, 108)
        
        while cap.isOpened():
            # Check if the window is closed (might break if something else has the same window name but it works so I'm not touching it)
            if cv2.getWindowProperty("Processed Frame", cv2.WND_PROP_VISIBLE) < 1:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Randomly decide whether or not to drop the frame (simulate network stutter)
            if random.random() < drop_rate:
                time.sleep(max_delay) # If the frame is dropped we just add the full delay 
                continue
            
            processed_frame = process_frame(frame, scale_factor, noise_level, compression_quality, original_dimensions)
            if processed_frame is not None:
                # Convert to RGB (cv2 uses BGR format)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Send to virtual camera
                cam.send(processed_frame_rgb)
                cam.sleep_until_next_frame()
                
                # Display the processed frame in the preview window thingy 
                cv2.imshow("Processed Frame", processed_frame)
                
                # Add random delay 
                time.sleep(random.uniform(0, max_delay))
            
            cv2.waitKey(1)
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cam(input("Enter mp4 path: "))
