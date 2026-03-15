# You can add helper functions here, e.g., FPS counter, image preprocessing, etc.

import time

class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0