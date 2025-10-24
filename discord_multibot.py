# discord_multibot.py
import os
import asyncio
import io
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, List

import discord
from discord.ext import commands
import requests
import yt_dlp
from gtts import gTTS

# Configuración desde variables de entorno
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

BOT_PREFIX = "!"
MAX_QUEUE_LENGTH = 50
TTS_LANGUAGE = "es"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("discord_multibot")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

@dataclass
class Song:
    url: str
    title: str
    requester_name: str
    channel: discord.TextChannel

class MusicQueue:
    def __init__(self, limit: int = MAX_QUEUE_LENGTH):
        self._queue: Deque[Song] = deque()
        self.limit = limit

    def enqueue(self, item: Song) -> bool:
        if len(self._queue) >= self.limit:
            return False
        self._queue.append(item)
        return True

    def dequeue(self) -> Optional[Song]:
        return self._queue.popleft() if self._queue else None

    def clear(self):
        self._queue.clear()

    def list_titles(self) -> List[str]:
        return [s.title for s in self._queue]

    def __len__(self):
        return len(self._queue)

music_queues: Dict[int, MusicQueue] = {}
conversation_history: Dict[str, List[dict]] = {}

YTDL_OPTS = {
    'format': 'bestaudio/best',
    'noplaylist': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'ytsearch',
    'extract_flat': 'in_playlist',
    'skip_download': True,
}

async def extract_info(search_or_url: str):
    loop = asyncio.get_event_loop()
    def _extract():
        with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
            return ydl.extract_info(search_or_url, download=False)
    return await loop.run_in_executor(None, _extract)

async def build_ffmpeg_source(video_url: str):
    before_options = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
    loop = asyncio.get_event_loop()
    def _direct_audio():
        with yt_dlp.YoutubeDL({'format': 'bestaudio/best', 'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info.get('url')
    direct_url = await loop.run_in_executor(None, _direct_audio)
    return discord.FFmpegOpusAudio(direct_url, before_options=before_options)

def add_to_history(context_key: str, role: str, content: str, max_len: int = 10):
    history = conversation_history.setdefault(context_key, [])
    history.append({'role': role, 'content': content})
    if len(history) > max_len:
        conversation_history[context_key] = history[-max_len:]

def deepseek_chat_response(context_key: str, user_prompt: str):
    add_to_history(context_key, 'user', user_prompt)
    payload = {
        "model": "gpt-4o",
        "messages": conversation_history[context_key],
        "max_tokens": 300,
        "temperature": 0.6,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        add_to_history(context_key, 'assistant', content)
        return content
    except Exception as e:
        log.exception("Error llamando a DeepSeek")
        return "Lo siento, ocurrió un error al solicitar la IA."

async def speak_text_in_voice(vc: discord.VoiceClient, text: str):
    if not vc or not vc.is_connected():
        return
    loop = asyncio.get_event_loop()

    def _generate_elevenlabs():
        try:
            voice_id = "pNInz6obpgDQGcFmaJgB"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
            payload = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError("Error en ElevenLabs")
            return io.BytesIO(response.content)
        except Exception as e:
            log.warning(f"Fallo ElevenLabs: {e}, usando gTTS")
            tts = gTTS(text, lang=TTS_LANGUAGE)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf

    audio_buf = await loop.run_in_executor(None, _generate_elevenlabs)
    temp_path = f"tts_{vc.guild.id}.mp3"
    with open(temp_path, "wb") as f:
        f.write(audio_buf.read())
    source = discord.FFmpegPCMAudio(temp_path)
    vc.play(source)
    while vc.is_playing():
        await asyncio.sleep(0.1)
    os.remove(temp_path)

@bot.event
async def on_ready():
    log.info(f"Bot conectado como {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if message.content.startswith(f"{BOT_PREFIX}ia"):
        prompt = message.content.replace(f"{BOT_PREFIX}ia", "").strip()
        await message.channel.trigger_typing()
        response = await asyncio.get_event_loop().run_in_executor(None, deepseek_chat_response, f"chan_{message.channel.id}", prompt)
        await message.channel.send(response)
        if message.guild.voice_client:
            await speak_text_in_voice(message.guild.voice_client, response)
    await bot.process_commands(message)

@bot.command(name="join")
async def join(ctx):
    if ctx.author.voice:
        await ctx.author.voice.channel.connect()
        await ctx.send(f"Conectado a {ctx.author.voice.channel.name}")
    else:
        await ctx.send("Únete a un canal de voz primero.")

@bot.command(name="leave")
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Desconectado.")
    else:
        await ctx.send("No estoy en un canal de voz.")

bot.run(DISCORD_TOKEN)
