"""A tiny stdio MCP server for tests/test_mcp_skills.py: `python tests/fixtures/mcp_server.py`.

Offers what the tests need and nothing else: a text result, a tool that fails on purpose, one that sleeps (timeout and
stop), one that reports its environment (proof the config's `env` reaches the child), one that reports its pid (proof
the child is gone after `close()`), one with an empty result, and one the test config does not list."""
import asyncio
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("fixture-books")

BOOKS = {"dune": "Dune by Frank Herbert, 1965", "emma": "Emma by Jane Austen, 1815"}


@server.tool()
def lookup_book(title: str) -> str:
    """Look up a book by its title."""
    try:
        return BOOKS[title.lower()]
    except KeyError:
        raise ToolError(f"no book called {title}") from None


@server.tool()
def add_todo(text: str) -> str:
    """Add an item to the to-do list."""
    return f"added: {text}"


@server.tool()
async def slow(seconds: float) -> str:
    """Wait, then answer."""
    await asyncio.sleep(seconds)
    return f"waited {seconds:g}"


@server.tool()
def secret_env(name: str) -> str:
    """Whether an environment variable is set in this process (never its value)."""
    return f"{name} is {'set' if name in os.environ else 'not set'}"


@server.tool()
def server_pid() -> str:
    """The process id of this server."""
    return str(os.getpid())


@server.tool()
def nothing() -> list:
    """Return no content."""
    return []


@server.tool()
def long_text(length: int) -> str:
    """A result of the given length."""
    return "x" * length


@server.tool()
def hidden_note() -> str:
    """A tool the test config does not list."""
    return "not offered"


if __name__ == "__main__":
    server.run("stdio")
