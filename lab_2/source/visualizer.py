import cv2

class FrameVisualizer:
    """Универсальная визуализация для всех режимов."""
    
    def __init__(self, window_name: str = "KLT tracking"):
        self.window_name = window_name
    
    def draw_tracking(self, frame, points, bbox, status_info=None):
        """Рисует трекинг на кадре."""
        x, y, w, h = bbox
        vis = frame.copy()
        
        # Цвет в зависимости от режима переинициализации
        color = (0, 255, 0)  # Зелёный по умолчанию
        if status_info and status_info.get("reinitialized"):
            color = (0, 0, 255)  # Красный при переинициализации
        
        # Рисуем точки
        for pt in points:
            a, b = pt.ravel().astype(int)
            cv2.circle(vis, (a, b), 3, color, -1)
        
        # Рисуем bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        # Основной текст
        cv2.putText(vis, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2)
        
        # Дополнительная информация
        if status_info:
            info_text = f"Mode: {status_info.get('mode', 'unknown')}"
            cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            points_info = f"Points: {status_info.get('points_count', 0)}"
            cv2.putText(vis, points_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            if status_info.get("new_points_count"):
                cv2.putText(vis, f"new_points: {status_info['new_points_count']}", 
                           (x + 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2)
        
        return vis
    
    def show_frame(self, frame):
        """Показывает кадр в окне."""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF == ord('q')
    
    @staticmethod
    def cleanup():
        """Закрывает все окна."""
        cv2.destroyAllWindows()
