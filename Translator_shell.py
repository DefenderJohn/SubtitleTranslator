from typing import List, Tuple
from tqdm import tqdm
import whisper
from transformers import AutoTokenizer, AutoModel

def transcribeToList(videoPath:str, model:str, showText:bool=False) -> List[Tuple[Tuple[float, float], str]]:
    _ = print("正在加载Whisper模型：") if showText else None
    model = whisper.load_model(model)
    _ = print("加载模型完成") if showText else None
    _ = print("开始转录：") if showText else None
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
        
    current = subtitles[index][1]
    forward = "\n"
    for forwardIndex in range(index, min(forwardCount + index, len(subtitles))):
        forward += subtitles[forwardIndex][1]
    return (history, current + forward + "\n这里是一段字幕，现在请你根据我们之前的对话，以及第一行字幕后给你的额外信息来翻译第一行字幕。请注意，只需要翻译第一行即可，后续的几行是提供给你作为上下文参考的。同样的，我们对话历史里已经翻译过的东西也不用再翻译一遍。")

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
            current_length = len(response)
    return translateResult

print("正在准备ChatGLM模型：")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
model = model.eval()
print("ChatGLM准备完毕")
transcribedSubtitles = transcribeToList(videoPath="test.ogg", model="large-v2", showText=True)
translatedSubtitles = []
historyCount = 10
forwardCount = 3
for index in tqdm(range(len(transcribedSubtitles)), desc="正在执行滑动窗口翻译", position=1):
    history, prompt = seperateSubtitleSegment(subtitles=transcribedSubtitles, translations=translatedSubtitles, index=index, historyCount=historyCount, forwardCount=forwardCount)
    translated = translateByChatGLM(history=history, prompt=prompt, model=model, tokenizer=tokenizer)
    beginTime = transcribedSubtitles[index][0][0]
    endTime = transcribedSubtitles[index][0][1]
    original = transcribedSubtitles[index][1]
    translatedSubtitles.append(((beginTime, endTime),(original, translated)))
for item in translatedSubtitles:
    print(item)