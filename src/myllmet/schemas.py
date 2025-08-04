from typing import TypedDict, List


class ChatMessage(TypedDict):
    role: str
    content: str
