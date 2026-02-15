# Protocol Bridge

A standalone gateway for translating between different agent communication standards (MCP, A2A, custom protocols) enabling heterogeneous multi-agent systems.

## Features

- **Multi-Protocol Support** - MCP, A2A, Custom, HTTP, and WebSocket protocols
- **Protocol Translation** - Seamlessly translate messages between different protocols
- **Adapter System** - Pluggable adapters for custom protocol implementations
- **Agent Connection Management** - Register, connect, and disconnect agents
- **Message Broadcasting** - Broadcast messages to multiple agents across protocols
- **Connection Statistics** - Track active connections and adapter usage

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from protocol_bridge import ProtocolGateway, GatewayProtocol, Agent, Protocol

# Create gateway
pg = ProtocolGateway()

# Register agents
pg.register_agent(Agent(agent_id="mcp-agent", name="MCP Agent", protocol=Protocol.MCP))
pg.register_agent(Agent(agent_id="a2a-agent", name="A2A Agent", protocol=Protocol.A2A))

# Connect agents
await pg.connect_agent("mcp-agent", GatewayProtocol.MCP)
await pg.connect_agent("a2a-agent", GatewayProtocol.A2A)

# Send message with translation
result = await pg.send_message(message, "a2a-agent", GatewayProtocol.A2A)
```

## Architecture

```
protocol-bridge/
├── src/
│   └── __init__.py
└── examples/
    └── demo.py
```

## License

MIT
