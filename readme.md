# cvoice

Cvoice is one tools for voice recognition and synthesis. which can change one pice of video audio from one person to another person.

## Dependencies

- ffmpeg
- pip install -r requirements.txt

## Issues

- the deepseek platform api website is closed, so we stop the model development. time : 2025-01-30.  

## TODO

- [x] use the log system to replace the print
- [ ] use the online model to replace the offline model
  - [x] use the deepseek r1 model to optimize the srt file is **accuracy**.
- [x] let the cli and args can do one small function, like the text to audio, audio to text, and so on.
- [ ] optimize the running logic, let the result video can be more accurate.

## Usage

the voice recognition example, which default output the .srt file on the `data` folder.

```shell
python cli.py transcribe --input data/extracted_audio.wav
```

## How it works?

- First, we need to split the audio from the video.
- Then we need to convert the audio to text.
- After that, we need to convert the text to another text. Let the text more accurate.
- Then we need to convert the text to audio.
- Finally, we need to merge the audio to the video.
