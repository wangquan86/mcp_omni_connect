import asyncio
from mcpomni_connect.cli import MCPClientCLI
from mcpomni_connect.client import MCPClient
from mcpomni_connect.llm import LLMConnection
from mcpomni_connect.utils import logger
from mcpomni_connect.config_manager import ConfigManager


async def async_main():
    try:
        # Initialize configuration
        config = ConfigManager()
        # Initialize client
        client = MCPClient(config, debug=True)
        # Initialize LLM connection
        llm_connection = LLMConnection(config)
        # Initialize CLI
        cli = MCPClientCLI(client, llm_connection, config)
        # Connect to servers
        await client.connect_to_servers()
        # Start chat loop
        await cli.chat_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info("Shutting down client...")
        if client:
            await client.cleanup()
        logger.info("Client shut down successfully")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
