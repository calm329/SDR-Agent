"""Main entry point for the SDR Agent."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from core.graph import create_sdr_workflow

# Load environment variables
load_dotenv()

# Initialize Typer app
app = typer.Typer(
    name="SDR Agent",
    help="AI-powered SDR Agent for company research and lead enrichment",
    add_completion=False
)

# Initialize Rich console
console = Console()


@app.command()
def run(
    query: str = typer.Argument(
        ...,
        help="Query for the SDR Agent (can be plain text or JSON with format specification)"
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file", "-i",
        help="Read query from a file instead of command line"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file", "-o",
        help="Save output to a file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed execution information"
    )
):
    """Run the SDR Agent with a query."""
    
    # Load query from file if specified
    if input_file:
        if not input_file.exists():
            console.print(f"[red]Error: Input file {input_file} not found[/red]")
            raise typer.Exit(1)
        
        with open(input_file, "r") as f:
            query = f.read().strip()
    
    # Show input
    if verbose:
        console.print(Panel(query, title="Input Query", border_style="blue"))
    
    # Create and run workflow
    try:
        with console.status("[bold green]Running SDR Agent...", spinner="dots"):
            workflow = create_sdr_workflow()
            result = asyncio.run(workflow.run(query))
        
        # Check if successful
        if result["success"]:
            # Display output
            output = result.get("formatted_output", result.get("output"))
            
            if isinstance(output, dict):
                # JSON output
                console.print(Panel(
                    JSON(json.dumps(output, indent=2)),
                    title="Result (JSON)",
                    border_style="green"
                ))
            else:
                # Plain text output
                console.print(Panel(
                    output,
                    title="Result (Plain Text)",
                    border_style="green"
                ))
            
            # Show citations
            if result.get("citations"):
                table = Table(title="Sources", show_header=True)
                table.add_column("Citation", style="cyan")
                for citation in result["citations"]:
                    table.add_row(citation)
                console.print(table)
            
            # Show execution info if verbose
            if verbose:
                console.print(f"\n[dim]Execution time: {result['execution_time']:.2f}s[/dim]")
                if result.get("errors"):
                    console.print("[yellow]Warnings:[/yellow]")
                    for error in result["errors"]:
                        console.print(f"  - {error}")
            
            # Save to file if specified
            if output_file:
                output_data = {
                    "query": query,
                    "result": output,
                    "citations": result.get("citations", []),
                    "execution_time": result["execution_time"]
                }
                
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                
                console.print(f"\n[green]Output saved to {output_file}[/green]")
        
        else:
            # Error occurred
            if result.get("timeout"):
                console.print(f"[red]Timeout Error: {result['error']}[/red]")
                console.print("\n[yellow]Tips:[/yellow]")
                console.print("  - The request may be too complex or websites may be blocking access")
                console.print("  - Try a more specific query or target a different company")
                console.print("  - Check your internet connection and Brightdata API credentials")
            else:
                console.print(f"[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def example(
    example_type: str = typer.Argument(
        "company",
        help="Type of example to show (company, contact, json)"
    )
):
    """Show example queries."""
    
    examples = {
        "company": {
            "query": "Tell me about Stripe",
            "description": "Basic company research query"
        },
        "contact": {
            "query": "Find the VP of Sales at Salesforce",
            "description": "Contact discovery query"
        },
        "json": {
            "query": json.dumps({
                "format": "json",
                "fields": {
                    "company_name": "string",
                    "industry": "string",
                    "headquarters": "string",
                    "description": "string"
                },
                "query": "Give me information about Stripe"
            }, indent=2),
            "description": "Structured JSON output request"
        }
    }
    
    if example_type in examples:
        example = examples[example_type]
        console.print(Panel(
            f"[bold]Description:[/bold] {example['description']}\n\n"
            f"[bold]Query:[/bold]\n{example['query']}",
            title=f"Example: {example_type}",
            border_style="blue"
        ))
    else:
        console.print(f"[red]Unknown example type: {example_type}[/red]")
        console.print("Available types: " + ", ".join(examples.keys()))


@app.command()
def test():
    """Run a simple test to verify the setup."""
    
    console.print("[bold]Running SDR Agent test...[/bold]\n")
    
    # Check environment variables
    env_vars = ["OPENAI_API_KEY", "BRIGHTDATA_API_KEY", "LANGSMITH_API_KEY"]
    all_set = True
    
    table = Table(title="Environment Variables", show_header=True)
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="green")
    
    for var in env_vars:
        import os
        if os.getenv(var):
            table.add_row(var, "✓ Set")
        else:
            table.add_row(var, "[red]✗ Not set[/red]")
            all_set = False
    
    console.print(table)
    
    if not all_set:
        console.print("\n[red]Some environment variables are missing![/red]")
        console.print("Please copy .env.example to .env and fill in your API keys.")
        raise typer.Exit(1)
    
    # Try a simple query
    console.print("\n[bold]Running test query...[/bold]")
    
    try:
        workflow = create_sdr_workflow()
        result = asyncio.run(workflow.run("Tell me about OpenAI"))
        
        if result["success"]:
            console.print("[green]✓ Test passed! The SDR Agent is working correctly.[/green]")
        else:
            console.print(f"[red]✗ Test failed: {result['error']}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Test failed with error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 