"""Protocol Bridge - Gateway for translating between different agent communication standards."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import uuid
import json


class AgentType(Enum):
    """Supported accelerator types for agents."""
    NVIDIA_GPU = "nvidia"
    AWS_TRAINIUM = "trainium"
    GOOGLE_TPU = "tpu"
    CPU = "cpu"


class Protocol(Enum):
    """Supported agent communication protocols."""
    MCP = "mcp"
    A2A = "a2a"
    CUSTOM = "custom"
    HTTP = "http"


class GatewayProtocol(Enum):
    """Supported protocols in the gateway."""
    MCP = "mcp"
    A2A = "a2a"
    CUSTOM = "custom"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class Agent:
    """Represents an agent in the system."""
    agent_id: str
    name: str
    agent_type: AgentType = AgentType.CPU
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    status: str = "idle"
    protocol: Protocol = Protocol.CUSTOM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "location": self.location,
            "status": self.status,
            "protocol": self.protocol.value
        }


@dataclass
class Message:
    """Represents a message between agents."""
    message_id: str
    sender: str
    receiver: str
    content: Any
    protocol: Protocol = Protocol.CUSTOM
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "protocol": self.protocol.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class EventEmitter:
    """Simple event emitter for state changes."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(*args, **kwargs)


@dataclass
class ProtocolAdapter:
    """Adapter for a specific protocol."""
    adapter_id: str
    protocol: GatewayProtocol
    name: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        raise NotImplementedError
    
    async def receive_message(self, raw_message: Any) -> Message:
        raise NotImplementedError
    
    async def connect(self, agent: Agent) -> bool:
        raise NotImplementedError
    
    async def disconnect(self, agent: Agent) -> bool:
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "protocol": self.protocol.value,
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class MCPPayload:
    """Model Context Protocol payload structure."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "params": self.params, "id": self.id}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPPayload':
        return cls(method=data.get("method", ""), params=data.get("params", {}), id=data.get("id"))


@dataclass
class A2APayload:
    """Agent-to-Agent Protocol payload structure."""
    action: str
    agent_id: str
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "agent_id": self.agent_id, "task": self.task, "context": self.context, "callback": self.callback}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'A2APayload':
        return cls(action=data.get("action", ""), agent_id=data.get("agent_id", ""), task=data.get("task", ""), context=data.get("context", {}), callback=data.get("callback"))


@dataclass
class TranslationResult:
    """Result of a protocol translation."""
    success: bool
    original_message: Optional[Message] = None
    translated_message: Optional[Message] = None
    source_protocol: Optional[GatewayProtocol] = None
    target_protocol: Optional[GatewayProtocol] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "original_message": self.original_message.to_dict() if self.original_message else None,
            "translated_message": self.translated_message.to_dict() if self.translated_message else None,
            "source_protocol": self.source_protocol.value if self.source_protocol else None,
            "target_protocol": self.target_protocol.value if self.target_protocol else None,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class GatewayConfig:
    """Configuration for the protocol gateway."""
    default_protocol: GatewayProtocol = GatewayProtocol.CUSTOM
    enable_translation: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_protocol": self.default_protocol.value,
            "enable_translation": self.enable_translation,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts
        }


class MCPAdapter(ProtocolAdapter):
    """Adapter for Model Context Protocol."""
    
    def __init__(self):
        super().__init__(adapter_id=str(uuid.uuid4()), protocol=GatewayProtocol.MCP, name="MCP Adapter", version="1.0")
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        mcp_payload = MCPPayload(method="agent.invoke", params={"target": target.agent_id, "content": message.content, "metadata": message.metadata}, id=message.message_id)
        print(f"[MCP] Sending: {json.dumps(mcp_payload.to_dict())}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        if isinstance(raw_message, dict):
            payload = MCPPayload.from_dict(raw_message)
            return Message(message_id=payload.id or str(uuid.uuid4()), sender=payload.params.get("source", "unknown"), receiver="local", content=payload.params.get("content", ""), protocol=Protocol.MCP, metadata=payload.params.get("metadata", {}))
        return Message(message_id=str(uuid.uuid4()), sender="unknown", receiver="local", content=str(raw_message), protocol=Protocol.MCP)
    
    async def connect(self, agent: Agent) -> bool:
        self._connections[agent.agent_id] = {"connected": True, "timestamp": datetime.now()}
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class A2AAdapter(ProtocolAdapter):
    """Adapter for Agent-to-Agent Protocol."""
    
    def __init__(self):
        super().__init__(adapter_id=str(uuid.uuid4()), protocol=GatewayProtocol.A2A, name="A2A Adapter", version="1.0")
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        a2a_payload = A2APayload(action="delegate", agent_id=target.agent_id, task=str(message.content), context=message.metadata)
        print(f"[A2A] Sending: {json.dumps(a2a_payload.to_dict())}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        if isinstance(raw_message, dict):
            payload = A2APayload.from_dict(raw_message)
            return Message(message_id=str(uuid.uuid4()), sender=payload.agent_id, receiver="local", content=payload.task, protocol=Protocol.A2A, metadata=payload.context)
        return Message(message_id=str(uuid.uuid4()), sender="unknown", receiver="local", content=str(raw_message), protocol=Protocol.A2A)
    
    async def connect(self, agent: Agent) -> bool:
        self._connections[agent.agent_id] = {"connected": True, "timestamp": datetime.now()}
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class CustomAdapter(ProtocolAdapter):
    """Adapter for custom/protocol-agnostic communication."""
    
    def __init__(self):
        super().__init__(adapter_id=str(uuid.uuid4()), protocol=GatewayProtocol.CUSTOM, name="Custom Adapter", version="1.0")
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        print(f"[Custom] Sending to {target.agent_id}: {message.content}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        return Message(message_id=str(uuid.uuid4()), sender="unknown", receiver="local", content=raw_message, protocol=Protocol.CUSTOM)
    
    async def connect(self, agent: Agent) -> bool:
        self._connections[agent.agent_id] = {"connected": True, "timestamp": datetime.now()}
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class HTTPAdapter(ProtocolAdapter):
    """Adapter for HTTP-based communication."""
    
    def __init__(self):
        super().__init__(adapter_id=str(uuid.uuid4()), protocol=GatewayProtocol.HTTP, name="HTTP Adapter", version="1.0")
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        print(f"[HTTP] POST to {target.agent_id}: {message.content}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        return Message(message_id=str(uuid.uuid4()), sender="http-source", receiver="local", content=raw_message, protocol=Protocol.HTTP)
    
    async def connect(self, agent: Agent) -> bool:
        self._connections[agent.agent_id] = {"connected": True, "timestamp": datetime.now()}
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class ProtocolGateway:
    """Gateway for translating between different agent communication protocols."""
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.adapters: Dict[GatewayProtocol, ProtocolAdapter] = {}
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.events = EventEmitter()
        self._message_cache: Dict[str, Any] = {}
        
        self.register_adapter(GatewayProtocol.MCP, MCPAdapter())
        self.register_adapter(GatewayProtocol.A2A, A2AAdapter())
        self.register_adapter(GatewayProtocol.CUSTOM, CustomAdapter())
        self.register_adapter(GatewayProtocol.HTTP, HTTPAdapter())
    
    def register_adapter(self, protocol: GatewayProtocol, adapter: ProtocolAdapter) -> None:
        self.adapters[protocol] = adapter
        self.events.emit("adapter_registered", protocol, adapter)
    
    def get_adapter(self, protocol: GatewayProtocol) -> Optional[ProtocolAdapter]:
        return self.adapters.get(protocol)
    
    def register_agent(self, agent: Agent) -> None:
        self.agents[agent.agent_id] = agent
        self.events.emit("agent_registered", agent)
    
    def unregister_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.connections.pop(agent_id, None)
            self.events.emit("agent_unregistered", agent)
    
    async def connect_agent(self, agent_id: str, protocol: Optional[GatewayProtocol] = None) -> bool:
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        protocol = protocol or self.config.default_protocol
        adapter = self.adapters.get(protocol)
        
        if not adapter:
            return False
        
        success = await adapter.connect(agent)
        
        if success:
            self.connections[agent_id] = {"protocol": protocol, "connected_at": datetime.now(), "adapter_id": adapter.adapter_id}
            self.events.emit("agent_connected", agent, protocol)
        
        return success
    
    async def disconnect_agent(self, agent_id: str) -> bool:
        if agent_id not in self.connections:
            return False
        
        connection = self.connections[agent_id]
        protocol = connection.get("protocol")
        
        adapter = self.adapters.get(protocol)
        agent = self.agents.get(agent_id)
        
        if adapter and agent:
            success = await adapter.disconnect(agent)
            if success:
                del self.connections[agent_id]
                self.events.emit("agent_disconnected", agent)
            return success
        
        return False
    
    async def send_message(self, message: Message, target_agent_id: str, target_protocol: Optional[GatewayProtocol] = None) -> TranslationResult:
        target_agent = self.agents.get(target_agent_id)
        if not target_agent:
            return TranslationResult(success=False, error=f"Agent {target_agent_id} not found")
        
        target_protocol = target_protocol or self._get_agent_protocol(target_agent_id)
        
        source_adapter = self.adapters.get(GatewayProtocol(message.protocol.value))
        target_adapter = self.adapters.get(target_protocol)
        
        if not source_adapter or not target_adapter:
            return TranslationResult(success=False, error="Protocol adapter not found")
        
        try:
            success = await target_adapter.send_message(message, target_agent)
            
            if success:
                self.events.emit("message_sent", message, target_agent)
                return TranslationResult(success=True, original_message=message, source_protocol=GatewayProtocol(message.protocol.value), target_protocol=target_protocol)
            else:
                return TranslationResult(success=False, error="Failed to send message")
        
        except Exception as e:
            return TranslationResult(success=False, error=str(e))
    
    async def translate_message(self, message: Message, target_protocol: GatewayProtocol) -> TranslationResult:
        source_protocol = GatewayProtocol(message.protocol.value)
        
        if source_protocol == target_protocol:
            return TranslationResult(success=True, original_message=message, translated_message=message, source_protocol=source_protocol, target_protocol=target_protocol)
        
        source_adapter = self.adapters.get(source_protocol)
        target_adapter = self.adapters.get(target_protocol)
        
        if not source_adapter or not target_adapter:
            return TranslationResult(success=False, error="Protocol adapter not found")
        
        try:
            translated_message = Message(message_id=str(uuid.uuid4()), sender=message.sender, receiver=message.receiver, content=message.content, protocol=Protocol(target_protocol.value), metadata={**message.metadata, "translated_from": source_protocol.value, "original_message_id": message.message_id})
            
            self.events.emit("message_translated", message, translated_message)
            
            return TranslationResult(success=True, original_message=message, translated_message=translated_message, source_protocol=source_protocol, target_protocol=target_protocol, metadata={"translation_time": datetime.now().isoformat()})
        
        except Exception as e:
            return TranslationResult(success=False, error=str(e))
    
    async def broadcast(self, message: Message, agent_ids: List[str], protocol: Optional[GatewayProtocol] = None) -> Dict[str, TranslationResult]:
        results = {}
        for agent_id in agent_ids:
            result = await self.send_message(message, agent_id, protocol)
            results[agent_id] = result
        return results
    
    def _get_agent_protocol(self, agent_id: str) -> GatewayProtocol:
        if agent_id in self.connections:
            return self.connections[agent_id].get("protocol", self.config.default_protocol)
        
        agent = self.agents.get(agent_id)
        if agent:
            return GatewayProtocol(agent.protocol.value)
        
        return self.config.default_protocol
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "registered_agents": len(self.agents),
            "active_connections": len(self.connections),
            "registered_adapters": len(self.adapters),
            "config": self.config.to_dict(),
            "connections": {agent_id: {"protocol": conn.get("protocol", "").value if conn.get("protocol") else "", "connected_at": conn.get("connected_at", "").isoformat() if conn.get("connected_at") else ""} for agent_id, conn in self.connections.items()}
        }
    
    def list_agents_by_protocol(self, protocol: GatewayProtocol) -> List[str]:
        return [agent_id for agent_id, conn in self.connections.items() if conn.get("protocol") == protocol]


__all__ = [
    "ProtocolGateway",
    "ProtocolAdapter",
    "MCPAdapter",
    "A2AAdapter",
    "CustomAdapter",
    "HTTPAdapter",
    "GatewayProtocol",
    "GatewayConfig",
    "TranslationResult",
    "MCPPayload",
    "A2APayload",
    "Agent",
    "Message",
    "EventEmitter",
    "AgentType",
    "Protocol",
]
