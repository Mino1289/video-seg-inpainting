from ultralytics import YOLO
import numpy as np
import cv2
import os
import collections
import time

class VideoPlayer:
    def __init__(self, source=0, flip=False, fps=None, skip_first_frames=0):
        self.source = source
        self.flip = flip
        self.fps = fps  # Not used for video files - they play at their natural rate
        self.skip_first_frames = skip_first_frames
        self.cap = None
        
    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {self.source}")
        
        for _ in range(self.skip_first_frames):
            self.cap.read()
            
    def next(self):
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        if self.flip:
            frame = cv2.flip(frame, 1)
        
        return frame
        
    def stop(self):
        if self.cap is not None:
            self.cap.release()
            
    def just_a_frame(self):
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        if self.flip:
            frame = cv2.flip(frame, 1)
            
        return frame

def run_instance_segmentation_with_inpainting(
    source=0,
    flip=False,
    use_popup=True,
    skip_first_frames=0,
    model=None,
    video_width=640,
    device="AUTO",
    save_output=False,
    output_path="output_video.mp4"
):
    player = None
    video_writer = None
    
    try:
        player = VideoPlayer(source=source, flip=flip, skip_first_frames=skip_first_frames)
        player.start()
        
        # Take a single frame to initialize background
        initial_frame = player.just_a_frame()
        if initial_frame is None:
            raise RuntimeError("Failed to capture a frame from the video source.")
        
        if video_width:
            scale = video_width / max(initial_frame.shape)
            initial_frame = cv2.resize(initial_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Initialize video writer if saving output
        if save_output:
            frame_height, frame_width = initial_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
            print(f"Saving output video to: {output_path}")
        
        background_img = initial_frame.copy().astype(np.float32)
        
        if use_popup:
            title = "Real-time Inpainting - Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        
        processing_times = collections.deque()
        actual_device = "AUTO (OpenVINO)" if device == "AUTO" else device
        print(f"Using device: {actual_device}")
        
        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            
            if video_width:
                scale = video_width / max(frame.shape)
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            input_image = np.array(frame)
            start_time = time.time()
            
            if device == "AUTO":
                detections = model(input_image, verbose=False, classes=[0], show_boxes=False)
            else:
                detections = model(input_image, verbose=False, device=f"intel:{device.lower()}" if device.lower() != "cpu" else "CPU", classes=[0], show_boxes=False)
            
            result = detections[0]
            
            current_frame = frame.astype(np.float32)
            
            person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            if result.masks is not None and len(result.masks.data) > 0:
                for i in range(result.masks.data.shape[0]):
                    mask = result.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    person_mask = np.logical_or(person_mask, binary_mask).astype(np.uint8)
                
                kernel = np.ones((15, 15), np.uint8)
                person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_DILATE, kernel)
            
            update_mask = (person_mask == 0)
            background_img[update_mask] = current_frame[update_mask]
            
            output_frame = frame.copy()
            
            if np.any(person_mask):
                background_uint8 = np.clip(background_img, 0, 255).astype(np.uint8)
                output_frame[person_mask == 1] = background_uint8[person_mask == 1]
                
            stop_time = time.time()            
            
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()
            
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time if processing_time > 0 else 0
            
            _, f_width = output_frame.shape[:2]
            font_scale = max(f_width / 1000, 0.5)
            
            cv2.putText(
                output_frame,
                f"Inference: {processing_time:.1f}ms ({fps:.1f} FPS) | Device: {actual_device}",
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
            
            # Display frame and handle input
            if use_popup:
                cv2.imshow(title, output_frame)
                key = cv2.waitKey(1) & 0xFF  # Minimal wait for key processing
                if key == 27 or key == ord('q'):  # ESC or Q to quit
                    break
                elif key == ord('r'):  # R to reset background
                    background_img = current_frame.copy()
                    print("Background reset!")
            else:
                # If no popup, just process without display delay
                pass
            
            if save_output and video_writer is not None:
                video_writer.write(output_frame)
    
    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        if player is not None:
            player.stop()
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_path}")
        if use_popup:
            cv2.destroyAllWindows()




# Run the demo
if __name__ == "__main__":
    model_path = "models/yolo11n-seg.pt"
    openvino_model_path = "models/yolo11n-seg_openvino_model_int8"
    device = "GPU"

    if not os.path.exists(openvino_model_path):
        # Basic conversion to OpenVINO format
        print("Converting YOLO model to OpenVINO format...")
        seg_model = YOLO(model_path)
        seg_model.export(format="openvino", dynamic=True, half=True, device=device)
        print("Model conversion completed!")

    print("Loading OpenVINO model...")
    seg_model = YOLO(openvino_model_path, task='segment')
    print("Model loaded successfully!")
    
    print("Starting real-time inpainting demo...")
    print("Press ESC or 'q' to quit")
    print("Press 'r' to reset background")
    
    source = 0
        
    run_instance_segmentation_with_inpainting(
        source=source,
        flip=False,
        use_popup=True,
        model=seg_model,
        video_width=640,
        device=device,
        save_output=True,
        output_path="output/processed_video.mp4"
    )