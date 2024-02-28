from typing import List, Tuple
from tqdm import tqdm
import whisper
from transformers import AutoTokenizer, AutoModel
import os
import torch

def transcribeToList(videoPath:str, modelInstance:str, showText:bool=False) -> List[Tuple[Tuple[float, float], str]]:
    _ = print("正在加载Whisper模型...") if showText else None
    modelInstance = whisper.load_model(modelInstance)
    _ = print("加载模型完成") if showText else None
    _ = print("开始转录：") if showText else None
    transcribeResult = modelInstance.transcribe(videoPath, language="en", verbose=False)
    _ = print("转录完成") if showText else None

    result = []
    for segment in transcribeResult["segments"]:
        text:str = segment["text"]
        startTime:float = segment["start"]
        endTime:float = segment["end"]
        result.append(((startTime, endTime), text))
    del modelInstance
    torch.cuda.empty_cache()
    return result

def seperateSubtitleSegment(subtitles:List[Tuple[Tuple[float, float], str]], translations:List[Tuple[Tuple[float, float], Tuple[str,str]]],
                             index:int, historyCount:int, forwardCount:int, additionalPrompt:str) -> Tuple[List,str]:
    history = []
    for historyIndex in range(max(index - historyCount, 0), index):
        userPrompt = {
            'role':'user',
            'content': buildOnePrompt(subtitles=subtitles, index=historyIndex, forwardCount=forwardCount, additionalPrompt=additionalPrompt)
        }
        assistanceRespond = {
            'role':'assistant',
            'content': translations[historyIndex][1][1]
        }
        history.append(userPrompt)
        history.append(assistanceRespond)
        
    return (history, buildOnePrompt(subtitles=subtitles, index=index, forwardCount=forwardCount, additionalPrompt=additionalPrompt))

def buildOnePrompt(subtitles:List[Tuple[Tuple[float, float], str]], index:int, forwardCount:int, additionalPrompt:str) -> str:
    current = '当前字幕：' + subtitles[index][1] + "\n"
    forward = "作为参考的后续字幕（不翻译）\n" if forwardCount != 0 else ""
    for forwardIndex in range(index, min(forwardCount + index, len(subtitles))):
        forward += subtitles[forwardIndex][1] + "\n"
    return current + forward + "\n" + additionalPrompt

def translateByChatGLM(history:str, prompt:str, model:AutoModel, tokenizer:AutoTokenizer) -> str:
    stop_stream = False
    past_key_values = None
    translateResult = ""
    current_length = 0
    successTranslate = False
    while not successTranslate:
        for response, history, past_key_values in model.stream_chat(tokenizer, prompt, history=history,
                                                                        past_key_values=past_key_values,
                                                                        return_past_key_values=True):
            if stop_stream:
                successTranslate = True
                break
            else:
                if len(translateResult) > 4096:
                    successTranslate = False
                    translateResult = ""
                    break
                translateResult += response[current_length:]
                current_length = len(response)
        if len(translateResult) < 4096:
            successTranslate = True
            break
    return translateResult

def convertTime(time:float) -> str:
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)

    return "{:02}:{:02}:{:02},{:03}".format(hours, minutes, seconds, milliseconds)

def writeToFile(content:str, fileName:str):
    with open(fileName, 'w', encoding='UTF-8') as file:
        file.write(content)

def findVideos(directory:str) -> List:
    videos = []
    fileExt = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in fileExt:
                videos.append(os.path.join(root, file))
    print(f"找到{len(videos)}个视频文件")
    return videos

def parse_config_to_variables(config_text):
    config_lines = config_text.split('\n')
    config_vars = {}

    for line in config_lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if value.isdigit():
                value = int(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            config_vars[key] = value
    return config_vars

with open('Config.ini', 'r') as file:
    config_vars = parse_config_to_variables(file.read())

WhisperModel = config_vars['WhisperModel']
ChatGLM_Model = config_vars['ChatGLM_Model']
batch = config_vars['MultiVideo']
Path = config_vars['Path']
ShowOutput = config_vars['ShowOutput']
historyCount = config_vars['HistoryCount']
forwardCount = config_vars['ForwardCount']
AdditionalPrompt = config_vars['AdditionalPrompt']
bothLanguage = config_vars['BothLanguage']
translate = config_vars['Translate']

if batch:
    selectedPath = Path
    for filePath in tqdm(findVideos(selectedPath), desc="正在翻译全部视频"):
        fileName = filePath.split('.')[0]
        transcribedSubtitles = transcribeToList(videoPath=filePath, modelInstance=WhisperModel, showText=ShowOutput)
        translatedSubtitles = []
        if translate:
            print("正在准备ChatGLM模型：")
            tokenizer = AutoTokenizer.from_pretrained(f"THUDM/{ChatGLM_Model}", trust_remote_code=True)
            model = AutoModel.from_pretrained(f"THUDM/{ChatGLM_Model}", trust_remote_code=True).cuda()
            model = model.eval()
            print("ChatGLM准备完毕")
            for index in tqdm(range(len(transcribedSubtitles)), desc="正在执行滑动窗口翻译", position=1):
                history, prompt = seperateSubtitleSegment(subtitles=transcribedSubtitles, translations=translatedSubtitles, index=index, historyCount=historyCount, forwardCount=forwardCount, additionalPrompt=AdditionalPrompt)
                translated = translateByChatGLM(history=history, prompt=prompt, model=model, tokenizer=tokenizer)
                beginTime = transcribedSubtitles[index][0][0]
                endTime = transcribedSubtitles[index][0][1]
                original = transcribedSubtitles[index][1]
                translatedSubtitles.append(((beginTime, endTime),(original, translated)))
            del model, tokenizer
            torch.cuda.empty_cache()
            index = 0
            result = ""
            for subtitle in translatedSubtitles:
                index += 1
                start = convertTime(subtitle[0][0])
                end = convertTime(subtitle[0][1])
                context = subtitle[1][1]
                original = subtitle[1][0]
                subtitleContent =f"{index}\n{start} --> {end}\n{context}\n{original}\n\n"
                result += subtitleContent
            writeToFile(fileName=f"{fileName}.srt", content=result)
        else:
            index = 0
            result = ""
            for subtitle in transcribedSubtitles:
                index += 1
                start = convertTime(subtitle[0][0])
                end = convertTime(subtitle[0][1])
                context = subtitle[1]
                subtitleContent = f"{index}\n{start} --> {end}\n{context}\n\n" if not bothLanguage else f"{index}\n{start} --> {end}\n{context}\n{original}\n\n"
                result += subtitleContent
            writeToFile(fileName=f"{fileName}.srt", content=result)
else:
    filePath = Path
    fileName = filePath.split('.')[0]
    transcribedSubtitles = transcribeToList(videoPath=filePath, modelInstance="large", showText=ShowOutput)
    translatedSubtitles = []
    if translate:
        print("正在准备ChatGLM模型：")
        tokenizer = AutoTokenizer.from_pretrained(f"THUDM/{ChatGLM_Model}", trust_remote_code=True)
        model = AutoModel.from_pretrained(f"THUDM/{ChatGLM_Model}", trust_remote_code=True).cuda()
        model = model.eval()
        print("ChatGLM准备完毕")
        for index in tqdm(range(len(transcribedSubtitles)), desc="正在执行滑动窗口翻译", position=1):
            history, prompt = seperateSubtitleSegment(subtitles=transcribedSubtitles, translations=translatedSubtitles, index=index, historyCount=historyCount, forwardCount=forwardCount,additionalPrompt=AdditionalPrompt)
            translated = translateByChatGLM(history=history, prompt=prompt, model=model, tokenizer=tokenizer)
            beginTime = transcribedSubtitles[index][0][0]
            endTime = transcribedSubtitles[index][0][1]
            original = transcribedSubtitles[index][1]
            translatedSubtitles.append(((beginTime, endTime),(original, translated)))
        del model
        torch.cuda.empty_cache()
        index = 0
        result = ""
        for subtitle in translatedSubtitles:
            index += 1
            start = convertTime(subtitle[0][0])
            end = convertTime(subtitle[0][1])
            original = subtitle[1][0]
            context = subtitle[1][1]
            subtitleContent =f"{index}\n{start} --> {end}\n{context}\n{original}\n\n"
            result += subtitleContent
        writeToFile(fileName=f"{fileName}.srt", content=result)
    else:
        index = 0
        result = ""
        for subtitle in transcribedSubtitles:
            index += 1
            start = convertTime(subtitle[0][0])
            end = convertTime(subtitle[0][1])
            context = subtitle[1]
            subtitleContent = f"{index}\n{start} --> {end}\n{context}\n\n"
            result += subtitleContent
        writeToFile(fileName=f"{fileName}.srt", content=result)