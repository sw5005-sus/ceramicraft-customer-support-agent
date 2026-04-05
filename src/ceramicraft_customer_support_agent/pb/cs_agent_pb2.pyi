from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ChatRequest(_message.Message):
    __slots__ = ("message", "thread_id", "auth_token")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    AUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    message: str
    thread_id: str
    auth_token: str
    def __init__(self, message: _Optional[str] = ..., thread_id: _Optional[str] = ..., auth_token: _Optional[str] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("reply", "thread_id")
    REPLY_FIELD_NUMBER: _ClassVar[int]
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    reply: str
    thread_id: str
    def __init__(self, reply: _Optional[str] = ..., thread_id: _Optional[str] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("thread_id",)
    THREAD_ID_FIELD_NUMBER: _ClassVar[int]
    thread_id: str
    def __init__(self, thread_id: _Optional[str] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("status", "message")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...
