import json
from contextlib import asynccontextmanager
from mcpomni_connect.client import MCPClient
from mcpomni_connect.config_manager import ConfigManager
from mcpomni_connect.llm import LLMConnection
from fastapi import FastAPI, Request
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    FileResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import datetime
from mcpomni_connect.constants import date_time_func
from mcpomni_connect.memory import (
    InMemoryShortTermMemory,
)
from mcpomni_connect.system_prompts import (
    generate_system_prompt,
)
from pydantic import BaseModel

from mcpomni_connect.agents.tool_calling_agent import ToolCallingAgent
from mcpomni_connect.agents.types import AgentConfig
from mcpomni_connect.utils import logger
from mcpomni_connect.tools import list_tools


class MCPClientWeb:
    def __init__(
        self, client: MCPClient, llm_connection: LLMConnection, config: ConfigManager
    ):
        self.client = client
        self.llm_connection = llm_connection
        self.agent_config = config.get("AgentConfig")
        self.MAX_CONTEXT_TOKENS = config.get("LLM", {}).get("max_context_length")
        self.MODE = {"auto": False, "orchestrator": False, "chat": True}
        self.client.debug = True
        self.in_memory_short_term_memory = InMemoryShortTermMemory(
            max_context_tokens=self.MAX_CONTEXT_TOKENS
        )

    async def handle_query(self, query: str, chat_id: str = None):

        tools = await list_tools(
            server_names=self.client.server_names,
            sessions=self.client.sessions,
        )
        if self.MODE["chat"]:
            # Generate system prompt for tool-supporting LLMs
            system_prompt = generate_system_prompt(
                current_date_time=date_time_func["format_date"](),
                available_tools=self.client.available_tools,
                llm_connection=self.llm_connection,
            )
            agent_config = AgentConfig(
                agent_name="tool_calling_agent",
                tool_call_timeout=self.agent_config.get("tool_call_timeout"),
                max_steps=self.agent_config.get("max_steps"),
                request_limit=self.agent_config.get("request_limit"),
                total_tokens_limit=self.agent_config.get("total_tokens_limit"),
                mcp_enabled=True,
            )
            tool_calling_agent = ToolCallingAgent(
                config=agent_config, debug=self.client.debug
            )
            response = await tool_calling_agent.run(
                query=query,
                chat_id=chat_id,
                system_prompt=system_prompt,
                llm_connection=self.llm_connection,
                sessions=self.client.sessions,
                server_names=self.client.server_names,
                tools_list=tools,
                available_tools=self.client.available_tools,
                add_message_to_history=(self.in_memory_short_term_memory.store_message),
                message_history=(self.in_memory_short_term_memory.get_messages),
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code executes before the application starts
    config = ConfigManager()
    app.state.client = MCPClient(config)
    app.state.client.debug = True
    app.state.llm_connection = LLMConnection(config)
    app.state.mcp_client_web = MCPClientWeb(
        client=app.state.client, llm_connection=app.state.llm_connection, config=config
    )
    logger.info("Initializing MCP client...")

    # Connect to servers
    await app.state.client.connect_to_servers()
    logger.info("MCP client initialized successfully")

    yield  # The application runs here

    # This code executes when the application is shutting down
    logger.info("Shutting down MCP client...")
    if app.state.client:
        await app.state.client.cleanup()
    logger.info("MCP client shut down successfully")


# Initialize FastAPI with the lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def root():
    return FileResponse(os.path.join(current_dir, "index.html"))


@app.get("/index.html")
async def index():
    return FileResponse(os.path.join(current_dir, "index.html"))


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy"})


# 挂载静态文件目录用于提供其他静态资源（如CSS、JS等）
app.mount("/static", StaticFiles(directory=current_dir), name="static")


def format_msg(usid, msg, meta, message_id, role="assistant"):
    response_message = {
        "message_id": message_id,
        "usid": usid,
        "role": role,
        "content": msg,
        "meta": meta,
        "likeordislike": None,
        "time": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    }
    return response_message


async def chat_endpoint(request: Request, user_input: str, chat_id: str):
    assistant_uuid = str(uuid.uuid4())
    try:
        response = await request.app.state.mcp_client_web.handle_query(
            query=user_input, chat_id=chat_id
        )
        yield (
            json.dumps(
                format_msg("ksa", response, [], assistant_uuid, "assistant")
            ).encode("utf-8")
            + b"\n"
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        yield (
            json.dumps(
                format_msg("ksa", str(e), [], str(uuid.uuid4()), "error")
            ).encode("utf-8")
            + b"\n"
        )


class ChatInput(BaseModel):
    query: str
    chat_id: str


@app.post("/chat/agent_chat")
async def chat(request: Request, chat_input: ChatInput):
    logger.info(f"Received query: {chat_input.query}")
    return StreamingResponse(
        chat_endpoint(
            request=request, user_input=chat_input.query, chat_id=chat_input.chat_id
        ),
        media_type="text/plain",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
