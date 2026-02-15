#!/usr/bin/env python3
"""Demo script for Protocol Bridge."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ProtocolGateway, GatewayProtocol, Agent, Message, Protocol

async def main():
    print("=" * 50)
    print("Protocol Bridge Demo")
    print("=" * 50)
    
    pg = ProtocolGateway()
    
    # Register agents
    pg.register_agent(Agent(agent_id="mcp-agent", name="MCP Agent", protocol=Protocol.MCP))
    pg.register_agent(Agent(agent_id="a2a-agent", name="A2A Agent", protocol=Protocol.A2A))
    pg.register_agent(Agent(agent_id="custom-agent", name="Custom Agent", protocol=Protocol.CUSTOM))
    
    # Connect agents
    await pg.connect_agent("mcp-agent", GatewayProtocol.MCP)
    await pg.connect_agent("a2a-agent", GatewayProtocol.A2A)
    await pg.connect_agent("custom-agent", GatewayProtocol.CUSTOM)
    
    # Send message with translation
    msg = Message(message_id="msg-1", sender="mcp-agent", receiver="a2a-agent", content="Hello!", protocol=Protocol.MCP)
    result = await pg.send_message(msg, "a2a-agent", GatewayProtocol.A2A)
    print(f"\nMessage sent: {result.success}")
    
    stats = pg.get_statistics()
    print(f"Registered agents: {stats['registered_agents']}")
    print(f"Active connections: {stats['active_connections']}")
    
    print("\n✓ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
