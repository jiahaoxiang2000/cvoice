import os

class FileHandler:
    @staticmethod
    def ensure_temp_dir():
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    @staticmethod
    def get_temp_path(filename):
        return os.path.join("temp", filename)
