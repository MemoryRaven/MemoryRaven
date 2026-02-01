#!/usr/bin/env python3
"""
Claude Memory CLI - Query and manage your memory
"""

import json
import os
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import Event, MemoryBridge
from .memory_os import MemoryOS, MemoryPolicy
from .retrieval import MemoryRetrieval

console = Console()


def get_memory_bridge():
    """Get memory bridge instance from environment or defaults"""
    return MemoryBridge(
        db_path=os.environ.get("CLAUDE_MEMORY_DB", "~/.claude_memory/memory.db"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        pinecone_environment=os.environ.get("PINECONE_ENVIRONMENT"),
        pinecone_index=os.environ.get("PINECONE_INDEX", "claude-memory"),
    )


@click.group()
def cli():
    """Claude Memory - Never forget anything"""
    pass


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, help="Number of results")
@click.option("--days", default=None, type=int, help="Search within last N days")
@click.option("--source", multiple=True, help="Filter by source")
@click.option("--type", "event_type", multiple=True, help="Filter by event type")
def search(query, limit, days, source, event_type):
    """Search your memories"""
    memory = get_memory_bridge()
    retrieval = MemoryRetrieval(memory)

    # Set time range if specified
    time_range = None
    if days:
        end = datetime.now().isoformat()
        start = (datetime.now() - timedelta(days=days)).isoformat()
        time_range = (start, end)

    # Search
    results = retrieval.search(
        query=query,
        limit=limit,
        time_range=time_range,
        sources=list(source) if source else None,
        event_types=list(event_type) if event_type else None,
    )

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display results
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Time", style="green", width=20)
    table.add_column("Source", style="blue", width=10)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Content", style="white", overflow="fold")

    for result in results:
        # Format time
        dt = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
        time_str = dt.strftime("%Y-%m-%d %H:%M")

        # Truncate content
        content = (
            result.content_text[:100] + "..."
            if len(result.content_text) > 100
            else result.content_text
        )

        table.add_row(f"{result.score:.3f}", time_str, result.source, result.event_type, content)

    console.print(table)


@cli.command()
@click.option("--hours", default=24, help="Show timeline for last N hours")
@click.option("--source", multiple=True, help="Filter by source")
def timeline(hours, source):
    """Show recent timeline of events"""
    memory = get_memory_bridge()
    retrieval = MemoryRetrieval(memory)

    # Calculate time range
    end = datetime.now().isoformat()
    start = (datetime.now() - timedelta(hours=hours)).isoformat()

    # Get timeline
    events = retrieval.get_timeline(
        start=start, end=end, sources=list(source) if source else None, limit=100
    )

    if not events:
        console.print("[yellow]No events in timeline[/yellow]")
        return

    # Group by hour
    current_hour = None

    for event in events:
        dt = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
        hour_str = dt.strftime("%Y-%m-%d %H:00")

        if hour_str != current_hour:
            current_hour = hour_str
            console.print(f"\n[bold blue]{hour_str}[/bold blue]")

        # Format event
        time_str = dt.strftime("%H:%M:%S")
        source_color = {
            "telegram": "green",
            "git": "yellow",
            "file": "cyan",
            "web": "blue",
            "decision": "red",
        }.get(event["source"], "white")

        console.print(
            f"  [{source_color}]{time_str}[/{source_color}] "
            f"[{source_color}]{event['source']}[/{source_color}] "
            f"{event['content_text'][:80]}"
        )


@cli.command()
@click.option("--source", default="cli", help="Event source")
@click.option("--type", "event_type", default="note", help="Event type")
def capture(source, event_type):
    """Manually capture a memory"""
    # Get input from user
    console.print("[bold]Enter your memory (Ctrl+D to finish):[/bold]")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass

    content = "\n".join(lines).strip()

    if not content:
        console.print("[red]No content provided[/red]")
        return

    # Capture
    memory = get_memory_bridge()
    event_id = memory.capture(
        Event(
            source=source,
            event_type=event_type,
            content_text=content,
            content_json={"text": content, "manual": True},
        )
    )

    if event_id:
        console.print(f"[green]✓ Captured memory: {event_id}[/green]")
    else:
        console.print("[yellow]Memory already exists (duplicate)[/yellow]")


@cli.command()
def stats():
    """Show memory statistics"""
    memory = get_memory_bridge()
    cursor = memory.conn.cursor()

    # Total events
    cursor.execute("SELECT COUNT(*) FROM events")
    total_events = cursor.fetchone()[0]

    # Events by source
    cursor.execute("""
        SELECT source, COUNT(*) as count 
        FROM events 
        GROUP BY source 
        ORDER BY count DESC
    """)
    by_source = cursor.fetchall()

    # Events by type
    cursor.execute("""
        SELECT event_type, COUNT(*) as count 
        FROM events 
        GROUP BY event_type 
        ORDER BY count DESC
    """)
    by_type = cursor.fetchall()

    # Recent activity
    cursor.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM events
        WHERE created_at > datetime('now', '-7 days')
        GROUP BY date
        ORDER BY date DESC
    """)
    recent_activity = cursor.fetchall()

    # Display stats
    console.print(Panel.fit(f"[bold cyan]Total Memories: {total_events:,}[/bold cyan]"))

    # By source table
    source_table = Table(title="Events by Source")
    source_table.add_column("Source", style="blue")
    source_table.add_column("Count", style="green")

    for source, count in by_source:
        source_table.add_row(source, f"{count:,}")

    console.print(source_table)

    # By type table
    type_table = Table(title="Events by Type")
    type_table.add_column("Type", style="magenta")
    type_table.add_column("Count", style="green")

    for event_type, count in by_type[:10]:  # Top 10
        type_table.add_row(event_type, f"{count:,}")

    console.print(type_table)

    # Recent activity
    console.print("\n[bold]Recent Activity (Last 7 Days)[/bold]")
    for date, count in recent_activity:
        bar_length = int(count / max(c for _, c in recent_activity) * 40)
        bar = "█" * bar_length
        console.print(f"{date}: {bar} {count}")


@cli.command()
@click.argument("event_id")
@click.option("--window", default=10, help="Context window size")
def context(event_id, window):
    """Show context around an event"""
    memory = get_memory_bridge()
    retrieval = MemoryRetrieval(memory)

    # Get context
    ctx = retrieval.get_context(event_id, window_size=window)

    if "error" in ctx:
        console.print(f"[red]{ctx['error']}[/red]")
        return

    # Display context
    console.print(Panel(f"Context for event {event_id}", style="bold"))

    for event in ctx["context"]:
        dt = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
        time_str = dt.strftime("%H:%M:%S")

        # Highlight the target event
        if event["event_id"] == event_id:
            console.print(
                f"[bold yellow]→ {time_str} [{event['source']}] {event['content_text']}[/bold yellow]"
            )
        else:
            console.print(f"  {time_str} [{event['source']}] {event['content_text']}")


@cli.command()
@click.option("--workspace", default=".", help="Clawdbot workspace root")
@click.option("--namespace", default="personal", help="Namespace tag to use (ns:<name>)")
def ingest_clawdbot(workspace, namespace):
    """Ingest Clawdbot memory markdown files (memory/*.md + MEMORY.md)"""
    memory = get_memory_bridge()
    memos = MemoryOS(memory, policy=MemoryPolicy(), workspace_root=workspace)
    res = memos.ingest_clawdbot_memory_files(namespace=namespace)
    console.print_json(json.dumps(res))


@cli.command()
@click.option("--workspace", default=".", help="Clawdbot workspace root")
def consolidate(workspace):
    """Create heuristic summaries for older long threads"""
    memory = get_memory_bridge()
    memos = MemoryOS(memory, policy=MemoryPolicy(), workspace_root=workspace)
    res = memos.consolidate()
    console.print_json(json.dumps(res))


@cli.command()
@click.argument("user_text")
@click.option("--limit", default=8, help="Max retrieved items")
def virtualize(user_text, limit):
    """Demo memory virtualization: prints injected memory block."""
    memory = get_memory_bridge()
    policy = MemoryPolicy()
    policy.inject_max_items = limit
    memos = MemoryOS(memory, policy=policy)
    msgs = [{"role": "user", "content": user_text}]
    augmented, meta = memos.virtualize(msgs)
    console.print(Panel.fit("Virtualized prompt (system injection shown below)", style="cyan"))
    for m in augmented:
        if m.get("role") == "system" and "MEMORY_CONTEXT" in (m.get("content") or ""):
            console.print(Panel(m["content"], style="green"))
    console.print(Panel.fit(json.dumps(meta, indent=2), title="meta"))


@cli.command()
@click.option("--days", default=30, help="Archive events older than N days")
@click.option("--dry-run", is_flag=True, help="Show what would be archived")
def archive(days, dry_run):
    """Archive old events to cold storage"""
    memory = get_memory_bridge()
    cursor = memory.conn.cursor()

    # Find events to archive
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

    cursor.execute(
        """
        SELECT COUNT(*) FROM events
        WHERE created_at < ?
    """,
        (cutoff_date,),
    )

    count = cursor.fetchone()[0]

    if count == 0:
        console.print("[yellow]No events to archive[/yellow]")
        return

    console.print(f"[blue]Found {count:,} events to archive (older than {days} days)[/blue]")

    if dry_run:
        console.print("[yellow]Dry run - no changes made[/yellow]")
        return

    # TODO: Implement actual archiving to S3/GitHub
    console.print("[red]Archiving not yet implemented[/red]")


if __name__ == "__main__":
    cli()
