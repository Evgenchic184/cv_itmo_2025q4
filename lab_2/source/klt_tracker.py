import cv2


class TrackingMode:
    BASIC = "basic"
    SKIP = "skip"


class KLTTracker:
    """Универсальный KLT трекер с выбором режима работы."""
    
    def __init__(self, frame_shape: tuple, mode: str = TrackingMode.BASIC, 
                 min_points_threshold: int = 50, reinitialize_ratio: float = 0.3):
        self.frame_width, self.frame_height = frame_shape
        self.mode = mode
        self.min_points_threshold = min_points_threshold
        self.reinitialize_ratio = reinitialize_ratio
        self.skipped = False
        
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=15,
        )
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=1 if mode == TrackingMode.BASIC else 3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        self.points = None
        self.old_gray = None
        self.initial_points_count = 0
        self.current_bbox = (0, 0, frame_shape[1], frame_shape[0])
    
    def initialize_tracking(self, first_gray_frame):
        """Инициализация трекинга на первом кадре."""
        self.old_gray = first_gray_frame.copy()
        roi = first_gray_frame
        
        self.points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        if self.points is None:
            raise ValueError("Нет ключевых точек в ROI")
        
        self.initial_points_count = len(self.points)
        self.is_reinitialized = False
        return self.points
    
    def track_next_frame(self, frame_gray):
        """Трекинг на следующем кадре."""
        if self.points is None:
            raise ValueError("Трекер не инициализирован")
        
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.points, None, **self.lk_params
        )
        
        if new_points is None:
            return None, None
        
        good_new = new_points[status == 1]
        current_points_count = len(good_new)
        
        flag = True
        
        if self.mode == TrackingMode.SKIP:
            lost_points_ratio = (self.initial_points_count - current_points_count) / self.initial_points_count
            needs_reinit = (lost_points_ratio > self.reinitialize_ratio or 
                           current_points_count < self.min_points_threshold)
            
            if needs_reinit:
                x, y, w, h = self._get_bbox(good_new.reshape(-1, 2))
                roi = frame_gray[y:y+h, x:x+w]
                new_features = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
                if new_features is not None and len(new_features) > len(good_new):
                    flag = False
        
        if flag:
            self.old_gray = frame_gray.copy()
            self.points = good_new.reshape(-1, 1, 2)
            self.current_bbox = self._get_bbox(good_new)
            self.skipped = True
        
        return good_new, {"skipped": self.skipped}
    
    def _get_bbox(self, points):
        """Получает bounding box по точкам."""
        if len(points) == 0:
            return 0, 0, self.frame_width, self.frame_height
        xs = points[:, 0]
        ys = points[:, 1]
        x_min = max(int(xs.min()), 0)
        y_min = max(int(ys.min()), 0)
        x_max = min(int(xs.max()), self.frame_width - 1)
        y_max = min(int(ys.max()), self.frame_height - 1)
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def get_bbox(self):
        """Возвращает текущий bounding box."""
        return self.current_bbox
    
    def get_current_points_count(self):
        """Текущее количество точек."""
        return len(self.points) if self.points is not None else 0
    
    def get_status_info(self):
        """Информация о состоянии трекера."""
        return {
            "mode": self.mode,
            "points_count": self.get_current_points_count(),
            "skipped": self.skipped,
            "initial_points_count": self.initial_points_count
        }
