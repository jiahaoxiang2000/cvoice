import os


class FileHandler:
    @staticmethod
    def ensure_temp_dir():
        temp_dir = "data"
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    @staticmethod
    def get_temp_path(filename):
        return os.path.join("data", filename)
