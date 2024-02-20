from typing import List, Tuple
from tqdm import tqdm
import whisper
from transformers import AutoTokenizer, AutoModel
import os

def transcribeToList(videoPath:str, model:str, showText:bool=False) -> List[Tuple[Tuple[float, float], str]]:
    _ = print("正在加载Whisper模型：") if showText else None
    model = whisper.load_model(model)
    _ = print("加载模型完成") if showText else None
    _ = print("开始转录：") if showText else None
    transcribeResult = model.transcribe(videoPath, verbose=False) if showText else model.transcribe(videoPath)
    _ = print("转录完成") if showText else None

    result = []
    for segment in transcribeResult["segments"]:
        text:str = segment["text"]
        startTime:float = segment["start"]
        endTime:float = segment["end"]
        result.append(((startTime, endTime), text))
    return result

def seperateSubtitleSegment(subtitles:List[Tuple[Tuple[float, float], str]], translations:List[Tuple[Tuple[float, float], Tuple[str,str]]],
                             index:int, historyCount:int, forwardCount:int) -> Tuple[List,str]:
    history = []
    for historyIndex in range(max(index - historyCount, 0), index):
        userPrompt = {
            'role':'user',
            'content': translations[historyIndex][1][0]
        }
        assistanceRespond = {
            'role':'assistant',
            'content': translations[historyIndex][1][1]
        }
        history.append(userPrompt)
        history.append(assistanceRespond)
        
    current = '当前字幕： ' + subtitles[index][1]
    forward = "作为参考的后续字幕（不翻译）\n"
    for forwardIndex in range(index, min(forwardCount + index, len(subtitles))):
        forward += subtitles[forwardIndex][1]
    return (history, current + forward + "\n翻译当前字幕")

def translateByChatGLM(history:str, prompt:str, model:AutoModel, tokenizer:AutoTokenizer) -> str:
    stop_stream = False
    past_key_values = None
    translateResult = ""
    current_length = 0
    for response, history, past_key_values in model.stream_chat(tokenizer, prompt, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
        if stop_stream:
            break
        else:
            translateResult += response[current_length:]
            print(response[current_length:], end="")
            current_length = len(response)
    print("")
    return translateResult

def convertTime(time:float) -> str:
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)

    return "{:02}:{:02}:{:02},{:03}".format(hours, minutes, seconds, milliseconds)

def writeToFile(content:str, fileName:str):
    with open(fileName, 'w') as file:
        file.write(content)

def findVideos(directory):
    videos = []
    fileExt = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in fileExt:
                videos.append(os.path.join(root, file))
    print(f"找到{len(videos)}个视频文件")
    return videos

selectedPath = "..."
for filePath in tqdm(findVideos(selectedPath), desc="正在翻译全部视频"):
    translate = False
    fileName = filePath.split('.')[0]
    transcribedSubtitles = transcribeToList(videoPath=filePath, model="large-v2", showText=True)
    translatedSubtitles = []
    historyCount = 10
    forwardCount = 3
    if translate:
        print("正在准备ChatGLM模型：")
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
        model = model.eval()
        print("ChatGLM准备完毕")
        for index in tqdm(range(len(transcribedSubtitles)), desc="正在执行滑动窗口翻译", position=1):
            history, prompt = seperateSubtitleSegment(subtitles=transcribedSubtitles, translations=translatedSubtitles, index=index, historyCount=historyCount, forwardCount=forwardCount)
            translated = translateByChatGLM(history=history, prompt=prompt, model=model, tokenizer=tokenizer)
            beginTime = transcribedSubtitles[index][0][0]
            endTime = transcribedSubtitles[index][0][1]
            original = transcribedSubtitles[index][1]
            translatedSubtitles.append(((beginTime, endTime),(original, translated)))
        for item in translatedSubtitles:
            print(item)
    index = 0
    result = ""
    for subtitle in transcribedSubtitles:
        index += 1
        start = convertTime(subtitle[0][0])
        end = convertTime(subtitle[0][1])
        context = subtitle[1]
        subtitleContent = f"{index}\n{start} --> {end}\n{context}\n\n"
        result += subtitleContent
    writeToFile(fileName=f"{selectedPath}/{fileName}.srt", content=result)
