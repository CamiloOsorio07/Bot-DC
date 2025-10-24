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

# =========================
# 🔐 Configuración (Tokens)
# =========================
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

BOT_PREFIX = "!"
MAX_QUEUE_LENGTH = 50
TTS_LANGUAGE = "es"

# =========================
# 🧠 Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("discord_multibot")

# =========================
# ⚙️ Intents y Bot
# =========================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

# =========================
# 🎵 Estructuras de datos
# =========================
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

# =========================
# 🔍 YouTube (yt_dlp)
# =========================
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

def is_url(string: str) -> bool:
    return string.startswith("http://") or string.startswith("https://")

async def build_ffmpeg_source(video_url: str):
    before_options = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
    loop = asyncio.get_event_loop()
    def _direct_audio():
        with yt_dlp.YoutubeDL({'format': 'bestaudio/best', 'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info.get('url')
    direct_url = await loop.run_in_executor(None, _direct_audio)
    return discord.FFmpegOpusAudio(direct_url, before_options=before_options)

# =========================
# 🤖 DeepSeek API (IA)
# =========================
def add_to_history(context_key: str, role: str, content: str, max_len: int = 10):
    history = conversation_history.setdefault(context_key, [])
    history.append({'role': role, 'content': content})
    if len(history) > max_len:
        conversation_history[context_key] = history[-max_len:]

def deepseek_chat_response(context_key: str, user_prompt: str, model: str = "gpt-4o"):
    add_to_history(context_key, 'user', user_prompt)
    payload = {
        "model": model,
        "messages": conversation_history[context_key],
        "max_tokens": 300,
        "temperature": 0.6,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
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

# =========================
# 🔊 TTS (voz)
# =========================
async def speak_text_in_voice(vc: discord.VoiceClient, text: str):
    if not vc or not vc.is_connected():
        return
    loop = asyncio.get_event_loop()

    def _generate_audio():
        try:
            voice_id = "pNInz6obpgDQGcFmaJgB"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError("Error en ElevenLabs")
            return io.BytesIO(response.content)
        except Exception:
            tts = gTTS(text, lang=TTS_LANGUAGE)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf

    audio_buf = await loop.run_in_executor(None, _generate_audio)
    temp_path = f"tts_{vc.guild.id}.mp3"
    with open(temp_path, "wb") as f:
        f.write(audio_buf.read())

    source = discord.FFmpegPCMAudio(temp_path)
    def _after_play(err):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    vc.play(source, after=_after_play)
    while vc.is_playing():
        await asyncio.sleep(0.1)

# =========================
# 🎶 Reproducción de música
# =========================
async def ensure_queue_for_guild(guild_id: int) -> MusicQueue:
    if guild_id not in music_queues:
        music_queues[guild_id] = MusicQueue(limit=MAX_QUEUE_LENGTH)
    return music_queues[guild_id]

async def start_playback_if_needed(guild: discord.Guild):
    vc = guild.voice_client
    if not vc or not vc.is_connected():
        return
    queue = music_queues.get(guild.id)
    if not queue or len(queue) == 0:
        return

    if not vc.is_playing():
        song = queue.dequeue()
        if not song:
            return
        try:
            source = await build_ffmpeg_source(song.url)
            def _after_play(err):
                coro = start_playback_if_needed(guild)
                asyncio.run_coroutine_threadsafe(coro, bot.loop)
            vc.play(source, after=_after_play)
            await song.channel.send(f"🎵 Reproduciendo ahora: **{song.title}** (pedido por {song.requester_name})")
        except Exception:
            await song.channel.send("Error al preparar el audio. Saltando...")

# =========================
# ⚡ Eventos del bot
# =========================
@bot.event
async def on_ready():
    log.info(f"✅ Bot conectado como {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.content.startswith(f"{BOT_PREFIX}ia") or bot.user.mentioned_in(message):
        prompt = message.content.replace(f"{BOT_PREFIX}ia", "").replace(f"<@{bot.user.id}>", "").strip()
        if not prompt:
            await message.channel.send("Dime qué quieres que responda.")
        else:
            await message.channel.trigger_typing()
            response = await asyncio.get_event_loop().run_in_executor(None, deepseek_chat_response, f"chan_{message.channel.id}", prompt)
            await message.channel.send(response)
            if message.guild.voice_client:
                await speak_text_in_voice(message.guild.voice_client, response)
    await bot.process_commands(message)

# =========================
# 🧩 Comandos
# =========================
@bot.command(name="join")
async def cmd_join(ctx):
    if ctx.author.voice and ctx.author.voice.channel:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Conectado a {channel.name}")
    else:
        await ctx.send("Únete a un canal de voz primero.")

@bot.command(name="leave")
async def cmd_leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        q = music_queues.get(ctx.guild.id)
        if q: q.clear()
        await ctx.send("Desconectado y cola vaciada.")
    else:
        await ctx.send("No estoy en un canal de voz.")

@bot.command(name="play")
async def cmd_play(ctx, *, search: str):
    if not search:
        await ctx.send("Especifica qué reproducir.")
        return
    if not ctx.voice_client:
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("Únete a un canal de voz primero.")
            return
        await ctx.author.voice.channel.connect()

    queue = await ensure_queue_for_guild(ctx.guild.id)
    query = search if is_url(search) else f"ytsearch:{search}"
    await ctx.send("🔎 Buscando en YouTube...")
    info = await extract_info(query)
    songs_added = 0

    if isinstance(info, dict) and 'entries' in info and info['entries']:
        for entry in info['entries']:
            url = entry.get('webpage_url') or entry.get('url')
            title = entry.get('title', 'Unknown title')
            song = Song(url, title, str(ctx.author), ctx.channel)
            if queue.enqueue(song):
                songs_added += 1
            else:
                break
    else:
        url = info.get('webpage_url') or info.get('url')
        title = info.get('title', 'Unknown title')
        song = Song(url, title, str(ctx.author), ctx.channel)
        if queue.enqueue(song):
            songs_added = 1

    await ctx.send(f"🎶 Añadidas {songs_added} canciones. (Cola: {len(queue)})")
    await start_playback_if_needed(ctx.guild)

@bot.command(name="skip")
async def cmd_skip(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("⏭ Canción saltada.")
    else:
        await ctx.send("No hay nada reproduciéndose.")

@bot.command(name="queue")
async def cmd_queue(ctx):
    q = music_queues.get(ctx.guild.id)
    if not q or len(q) == 0:
        await ctx.send("La cola está vacía.")
        return
    titles = q.list_titles()
    show = titles[:20]
    formatted = "\n".join([f"{i+1}. {t}" for i, t in enumerate(show)])
    await ctx.send(f"📜 Cola actual:\n{formatted}\nTotal: {len(q)}")

@bot.command(name="stop")
async def cmd_stop(ctx):
    if ctx.voice_client:
        ctx.voice_client.stop()
        q = music_queues.get(ctx.guild.id)
        if q: q.clear()
        await ctx.send("🛑 Reproducción detenida y cola vaciada.")
    else:
        await ctx.send("No estoy en canal de voz.")

# =========================
# 🚀 Ejecutar bot
# =========================
bot.run(DISCORD_TOKEN)
