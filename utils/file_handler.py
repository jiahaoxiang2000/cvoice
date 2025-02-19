import os
import tempfile

class FileHandler:
    TEMP_DIR = None

    @classmethod
    def ensure_temp_dir(cls):
        if not cls.TEMP_DIR:
            cls.TEMP_DIR = tempfile.mkdtemp()
        return cls.TEMP_DIR

    @classmethod
    def get_temp_path(cls, filename):
        temp_dir = cls.ensure_temp_dir()
        return os.path.join(temp_dir, filename)
