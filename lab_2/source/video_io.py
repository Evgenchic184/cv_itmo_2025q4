import cv2

class VideoReader:
    """Чтение видео с инициализацией первого кадра."""
    
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def read_first_frame(self):
        """Читает первый кадр и возвращает его."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Не удалось открыть видео")
        return frame
    
    def read_next_frame(self):
        """Читает следующий кадр."""
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Освобождает ресурсы."""
        self.cap.release()


class VideoWriter:
    """Запись видео с заданными параметрами."""
    
    def __init__(self, output_path: str, frame_shape: tuple, fps: float):
        self.frame_width, self.frame_height = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.frame_width, self.frame_height)
        )
    
    def write_frame(self, frame):
        """Записывает кадр в видео."""
        self.writer.write(frame)
    
    def release(self):
        """Освобождает ресурсы."""
        self.writer.release()
