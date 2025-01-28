from setuptools import setup, find_packages

setup(
    name="cvoice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "moviepy>=1.0.3",
        "SpeechRecognition>=3.8.1",
        "pyttsx3>=2.90",
        "transformers>=4.20.0",
        "torch>=1.9.0",
    ],
    entry_points={
        "console_scripts": [
            "cvoice=cvoice.cli:main",
        ],
    },
    python_requires=">=3.6",
)
