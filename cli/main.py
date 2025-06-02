import asyncio
import click
import json
import os
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.text import Text

from shared.message_bus import MessageBus, Channels, get_message_bus
from shared.models import (
    AgentMessage, EventType, InterviewContext, InterviewState, 
    Difficulty, Message
)

console = Console()

class InterviewCLI:
    def __init__(self):
        self.message_bus: Optional[MessageBus] = None
        self.current_session: Optional[str] = None
        self.context: Optional[InterviewContext] = None
        self.listening = False
        
    async def initialize(self):
        """Initialize CLI and message bus connection"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.message_bus = await get_message_bus(redis_url)
        
        # Subscribe to user interaction channel
        self.message_bus.subscribe(Channels.USER_INTERACTION, self.handle_agent_response)
        console.print("[green]✓[/green] Connected to interview system")
        
    async def handle_agent_response(self, message: AgentMessage):
        """Handle responses from agents"""
        if message.target_agent == "user" or message.target_agent == "cli":
            payload = message.payload
            speaker = payload.get("speaker", message.source_agent)
            content = payload.get("content", "")
            
            if speaker == "interviewer":
                console.print(Panel(content, title="[blue]Interviewer[/blue]", border_style="blue"))
            elif speaker == "system":
                console.print(f"[yellow]System:[/yellow] {content}")
            else:
                console.print(f"[cyan]{speaker}:[/cyan] {content}")
    
    async def start_interview(self, difficulty: str = "medium", user_id: str = "test_user"):
        """Start a new interview session"""
        try:
            # Generate session ID
            import uuid
            self.current_session = str(uuid.uuid4())
            
            # Create initial context
            self.context = InterviewContext(
                session_id=self.current_session,
                user_id=user_id,
                interview_state=InterviewState.INITIALIZING
            )
            
            # Store context in message bus
            await self.message_bus.store_context(self.current_session, self.context)
            
            # Send start interview message to coordinator
            start_message = AgentMessage(
                event_type=EventType.INTERVIEW_STATE_CHANGE,
                source_agent="cli",
                target_agent="coordinator",
                payload={
                    "action": "start_interview",
                    "session_id": self.current_session,
                    "user_id": user_id,
                    "difficulty": difficulty
                },
                context_snapshot=self.context.to_dict()
            )
            
            await self.message_bus.publish(Channels.COORDINATION, start_message)
            
            console.print(Panel(
                f"[green]Interview Started![/green]\n"
                f"Session ID: {self.current_session}\n"
                f"Difficulty: {difficulty.upper()}\n"
                f"User: {user_id}",
                title="Session Info"
            ))
            
            # Start listening for responses
            if not self.listening:
                self.listening = True
                asyncio.create_task(self.message_bus.start_listening())
                
        except Exception as e:
            console.print(f"[red]Error starting interview: {e}[/red]")
    
    async def send_user_input(self, content: str):
        """Send user input to the system"""
        if not self.current_session:
            console.print("[red]No active interview session. Start an interview first.[/red]")
            return
            
        try:
            # Update context
            if self.context:
                self.context.add_conversation("user", content)
                await self.message_bus.store_context(self.current_session, self.context)
            
            # Send message
            user_message = AgentMessage(
                event_type=EventType.USER_INPUT,
                source_agent="user",
                target_agent="coordinator",
                payload={
                    "content": content,
                    "session_id": self.current_session,
                    "timestamp": datetime.now().isoformat()
                },
                context_snapshot=self.context.to_dict() if self.context else {}
            )
            
            await self.message_bus.publish(Channels.USER_INTERACTION, user_message)
            console.print(f"[green]You:[/green] {content}")
            
        except Exception as e:
            console.print(f"[red]Error sending input: {e}[/red]")
    
    async def submit_code(self, code: str):
        """Submit code for analysis and execution"""
        if not self.current_session:
            console.print("[red]No active interview session.[/red]")
            return
            
        try:
            # Update context
            if self.context:
                self.context.update_code(code)
                await self.message_bus.store_context(self.current_session, self.context)
            
            # Send code change message
            code_message = AgentMessage(
                event_type=EventType.CODE_CHANGE,
                source_agent="user",
                target_agent="code_analyzer",
                payload={
                    "code": code,
                    "session_id": self.current_session,
                    "action": "analyze"
                },
                context_snapshot=self.context.to_dict() if self.context else {}
            )
            
            await self.message_bus.publish(Channels.CODE_ANALYSIS, code_message)
            
            # Display code with syntax highlighting
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="[green]Code Submitted[/green]"))
            
        except Exception as e:
            console.print(f"[red]Error submitting code: {e}[/red]")
    
    async def execute_code(self):
        """Request code execution"""
        if not self.current_session or not self.context or not self.context.current_code:
            console.print("[red]No code to execute.[/red]")
            return
            
        try:
            execution_message = AgentMessage(
                event_type=EventType.EXECUTION_REQUEST,
                source_agent="user",
                target_agent="execution",
                payload={
                    "code": self.context.current_code,
                    "session_id": self.current_session,
                    "test_mode": True
                },
                context_snapshot=self.context.to_dict()
            )
            
            await self.message_bus.publish(Channels.EXECUTION, execution_message)
            console.print("[yellow]Executing code...[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error executing code: {e}[/red]")
    
    async def get_status(self):
        """Get current interview status"""
        if not self.current_session:
            console.print("[red]No active interview session.[/red]")
            return
            
        try:
            # Get latest context
            context_data = await self.message_bus.get_context(self.current_session)
            if not context_data:
                console.print("[red]Session context not found.[/red]")
                return
            
            # Get active agents
            active_agents = await self.message_bus.get_active_agents()
            
            # Create status table
            table = Table(title="Interview Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Session ID", self.current_session)
            table.add_row("State", context_data.get("interview_state", "unknown"))
            table.add_row("Problem", context_data.get("current_problem", {}).get("title", "None"))
            table.add_row("Code Changes", str(context_data.get("performance_metrics", {}).get("code_changes", 0)))
            table.add_row("Questions Asked", str(context_data.get("performance_metrics", {}).get("questions_asked", 0)))
            table.add_row("Active Agents", ", ".join(active_agents))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error getting status: {e}[/red]")
    
    async def end_interview(self):
        """End the current interview"""
        if not self.current_session:
            console.print("[red]No active interview session.[/red]")
            return
            
        try:
            end_message = AgentMessage(
                event_type=EventType.INTERVIEW_STATE_CHANGE,
                source_agent="user",
                target_agent="coordinator",
                payload={
                    "action": "end_interview",
                    "session_id": self.current_session
                },
                context_snapshot=self.context.to_dict() if self.context else {}
            )
            
            await self.message_bus.publish(Channels.COORDINATION, end_message)
            console.print("[green]Interview ended.[/green]")
            
            self.current_session = None
            self.context = None
            
        except Exception as e:
            console.print(f"[red]Error ending interview: {e}[/red]")

# CLI Commands
cli = InterviewCLI()

@click.group()
def main():
    """AI Technical Interview CLI"""
    pass

@main.command()
@click.option('--difficulty', default='medium', type=click.Choice(['easy', 'medium', 'hard']))
@click.option('--user-id', default='test_user', help='User identifier')
def start(difficulty, user_id):
    """Start a new interview session"""
    async def _start():
        await cli.initialize()
        await cli.start_interview(difficulty, user_id)
        # Keep CLI running for interaction
        await asyncio.sleep(2)  # Give time for initial messages
        console.print("\n[yellow]Interview started! Use 'input', 'code', 'execute', 'status', or 'end' commands.[/yellow]")
    
    asyncio.run(_start())

@main.command()
@click.argument('message')
def input(message):
    """Send input to the interviewer"""
    async def _input():
        if not cli.message_bus:
            await cli.initialize()
        await cli.send_user_input(message)
        await asyncio.sleep(1)  # Wait for response
    
    asyncio.run(_input())

@main.command()
@click.argument('code')
def code(code):
    """Submit code for analysis"""
    async def _code():
        if not cli.message_bus:
            await cli.initialize()
        await cli.submit_code(code)
        await asyncio.sleep(1)
    
    asyncio.run(_code())

@main.command()
def execute():
    """Execute submitted code"""
    async def _execute():
        if not cli.message_bus:
            await cli.initialize()
        await cli.execute_code()
        await asyncio.sleep(2)  # Wait for execution results
    
    asyncio.run(_execute())

@main.command()
def status():
    """Get interview status"""
    async def _status():
        if not cli.message_bus:
            await cli.initialize()
        await cli.get_status()
    
    asyncio.run(_status())

@main.command()
def end():
    """End the current interview"""
    async def _end():
        if not cli.message_bus:
            await cli.initialize()
        await cli.end_interview()
    
    asyncio.run(_end())

@main.command()
def interactive():
    """Start interactive mode"""
    async def _interactive():
        await cli.initialize()
        console.print("[green]Interactive mode started. Type 'help' for commands.[/green]")
        
        while True:
            try:
                command = console.input("\n[bold blue]> [/bold blue]")
                
                if command.lower() in ['exit', 'quit']:
                    break
                elif command.lower() == 'help':
                    console.print("""
[yellow]Available commands:[/yellow]
- start [difficulty] [user_id]: Start interview
- input <message>: Send message to interviewer  
- code <code>: Submit code
- execute: Execute current code
- status: Show interview status
- end: End interview
- exit/quit: Exit interactive mode
                    """)
                elif command.startswith('start'):
                    parts = command.split()
                    difficulty = parts[1] if len(parts) > 1 else 'medium'
                    user_id = parts[2] if len(parts) > 2 else 'test_user'
                    await cli.start_interview(difficulty, user_id)
                elif command.startswith('input '):
                    message = command[6:]  # Remove 'input '
                    await cli.send_user_input(message)
                elif command.startswith('code '):
                    code = command[5:]  # Remove 'code '
                    await cli.submit_code(code)
                elif command == 'execute':
                    await cli.execute_code()
                elif command == 'status':
                    await cli.get_status()
                elif command == 'end':
                    await cli.end_interview()
                else:
                    console.print("[red]Unknown command. Type 'help' for available commands.[/red]")
                    
                await asyncio.sleep(0.5)  # Brief pause for responses
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print("[green]Goodbye![/green]")
    
    asyncio.run(_interactive())

if __name__ == '__main__':
    main()