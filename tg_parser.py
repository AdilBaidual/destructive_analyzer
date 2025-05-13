import os
import pandas as pd
import datetime
import asyncio
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetMessagesRequest
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.getenv("TG_API_ID"))
api_hash = os.getenv("TG_API_HASH")


async def fetch_messages(channel_name: str, post_count: int) -> str:
    async with TelegramClient("session", api_id, api_hash) as client:
        messages = []
        timestamps = []

        async for message in client.iter_messages(channel_name, limit=post_count):
            if message.text:
                messages.append(message.text)
                timestamps.append(message.date)

        df = pd.DataFrame({
            "text": messages,
            "created_at": timestamps
        })

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{channel_name.strip('@')}.csv"
        filepath = os.path.join("raw_data", filename)
        df.to_csv(filepath, index=False)

        return filename

def parse_telegram_channel(channel_name: str, post_count: int) -> str:
    return asyncio.run(fetch_messages(channel_name, post_count))

async def fetch_single_post(channel_name: str, post_id: int) -> str:
    async with TelegramClient("session", api_id, api_hash) as client:
        entity = await client.get_entity(channel_name)
        message = await client.get_messages(entity, ids=post_id)

        if not message or not message.text:
            raise ValueError("Сообщение не найдено или не содержит текста.")

        df = pd.DataFrame({
            "text": [message.text],
            "created_at": [message.date]
        })

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{channel_name.strip('@')}_post_{post_id}.csv"
        filepath = os.path.join("raw_data", filename)
        df.to_csv(filepath, index=False)

        return filename

def parse_single_post(channel_name: str, post_id: int) -> str:
    return asyncio.run(fetch_single_post(channel_name, post_id))
