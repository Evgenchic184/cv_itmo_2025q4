from video_io import VideoReader, VideoWriter
from klt_tracker import KLTTracker, TrackingMode
from visualizer import FrameVisualizer
import cv2

VIDEO_PATH = "test-videos/mona-lisa-blur-extra-credit.avi"
OUTPUT_PATH = "./results/mona-lisa-blur-extra-credit-skip.mp4"

def main(mode: str = TrackingMode.BASIC):
    reader = VideoReader(VIDEO_PATH)
    first_frame = reader.read_first_frame()
    frame_height, frame_width = first_frame.shape[:2]
    
    writer = VideoWriter(OUTPUT_PATH, (frame_width, frame_height), reader.fps)
    tracker = KLTTracker((frame_width, frame_height), mode=mode)
    visualizer = FrameVisualizer(f"KLT tracking ({mode})")
    
    try:
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        tracker.initialize_tracking(first_gray)
        
        while True:
            ret, frame = reader.read_next_frame()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            good_points, status_info = tracker.track_next_frame(frame_gray)
            
            if good_points is None:
                break
            
            bbox = tracker.get_bbox()
            full_status = tracker.get_status_info()
            full_status.update(status_info)
            
            vis_frame = visualizer.draw_tracking(frame, good_points, bbox, full_status)
            
            writer.write_frame(vis_frame)
            
            if visualizer.show_frame(vis_frame):
                break
        
        print(f"Готово ({mode}), сохранено в", OUTPUT_PATH)
        
    finally:
        reader.release()
        writer.release()
        visualizer.cleanup()

if __name__ == "__main__":
    # main(TrackingMode.BASIC)
    
    main(TrackingMode.SKIP)
