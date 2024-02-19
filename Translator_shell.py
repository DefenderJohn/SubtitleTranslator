from typing import List, Tuple
import whisper

def transcribeToList(videoPath:str, model:str, showText:bool=False) -> List[Tuple[Tuple[float, float], str]]:
    _ = print("模型加载中") if showText else None
    model = whisper.load_model(model)
    _ = print("加载模型完成") if showText else None
    _ = print("开始转录") if showText else None
    transcribeResult = model.transcribe(videoPath, verbose=False) if showText else model.transcribe(videoPath)
    _ = print("转录完成，文本为：") if showText else None

    result = []
    for segment in transcribeResult["segments"]:
        text:str = segment["text"]
        startTime:float = segment["start"]
        endTime:float = segment["end"]
        result.append(((startTime, endTime), text))
    return result

def seperateSubtitleSegment(subtitles:List[Tuple[Tuple[float, float], str]], translations:List[Tuple[Tuple[float, float], Tuple[str,str]]],
                             index:int, historyCount:int, forwardCount:int) -> Tuple[str,str]:
    history = ""
    for historyIndex in range(max(index - historyCount, 0), index):
        history += translations[historyIndex][1][0] + "\n" + translations[historyIndex][1][1] + "\n"
    current = subtitles[index][1]
    forward = "\n"
    for forwardIndex in range(index, min(forwardCount + index, len(subtitles))):
        forward += subtitles[forwardIndex][1]
    return (history, current + forward)

subtitles = transcribeToList(videoPath="test1.mp4", model="large-v2", showText=True)
for subtitle in subtitles:
    print(subtitle)
subtitles