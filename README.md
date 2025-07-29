# SDR Agent - AI-Powered Sales Development Representative

An intelligent, multi-agent system built with LangGraph that automates sales development tasks. This single-turn agent provides comprehensive company research, contact discovery, lead qualification, and personalized outreach suggestions - all with real-time data and transparent source citations.

## 🚀 Key Features

### Core Capabilities
-   **Multi-Agent Architecture**: Orchestrated workflow with specialized agents for research, contact discovery, qualification, and personalization
-   **Real-Time Data Access**: Live web scraping and search through BrightData MCP integration
-   **Dynamic Model Switching**: Automatically selects between `gpt-4o` (simple queries) and `o4-mini` (complex reasoning) based on query complexity
-   **100% LangSmith Integration**: All prompts managed exclusively through LangSmith for version control and A/B testing
-   **Flexible Output**: Supports both structured JSON and human-readable text formats
-   **Comprehensive Citations**: Every piece of information includes transparent source attribution

### Technical Excellence
-   **Token Efficiency**: Smart content truncation (3KB limit) and parallel agent execution
-   **Robust Error Handling**: Graceful degradation with fallback strategies
-   **Full Observability**: Complete LangSmith tracing for debugging and optimization
-   **No Hallucinations**: Returns "information not available" rather than making up data
-   **Production-Ready**: Clean architecture with comprehensive error handling

## 🏗️ Architecture Design & Justification

### Multi-Agent System
The SDR Agent uses a LangGraph-based orchestration pattern with specialized agents:

1. **Router Agent**: Analyzes queries and determines optimal execution path
2. **Company Research Agent**: Gathers comprehensive company intelligence
3. **Contact Discovery Agent**: Finds decision-makers with email enrichment
4. **Lead Qualification Agent**: Scores prospects based on fit and signals
5. **Outreach Personalization Agent**: Creates tailored messaging strategies
6. **Output Formatter Agent**: Ensures consistent, citation-rich outputs

### Why This Architecture?
- **Modularity**: Each agent can be independently optimized and tested
- **Scalability**: Easy to add new agents or data sources
- **Efficiency**: Parallel execution where possible, sequential where necessary
- **Maintainability**: Clear separation of concerns

## 📊 Token Efficiency & Prompt Optimization

### Optimization Strategies
1. **Smart Truncation**: All scraped content limited to 3KB before LLM processing
2. **Parallel Execution**: Agents run concurrently when dependencies allow
3. **Dynamic Model Selection**: 
   - Simple queries → `gpt-4o` (temperature=0)
   - Complex queries → `o4-mini` (temperature=1)
4. **Prompt Management**: All prompts in LangSmith for easy optimization without code changes

### Model Switching Logic
The router detects complexity based on:
- Multi-entity comparisons
- Analysis requests
- Specific role mentions
- Current events queries
- Multiple task requirements

## 🎯 Agent Accuracy & SDR-Helpful Response Quality

### Data Sources
- **Primary**: LinkedIn company/people data via BrightData
- **Secondary**: Company websites, news articles, search results
- **Enrichment**: Email pattern detection and validation

### Quality Assurance
- LLM-based extraction (no brittle regex)
- Multi-source validation
- Confidence scoring for contacts
- Real-time data prioritization

## 📌 Structured & Text Output Flexibility

The agent supports multiple output formats:
```json
{
  "query": "Your question",
  "format": "json",  // or "text"
  "fields": {        // optional field specification
    "company": true,
    "contacts": true
  }
}
```

## 📖 Citation Completeness & Accuracy

Every response includes:
- Direct source links
- Data freshness indicators
- Confidence levels for enriched data
- Clear attribution for all claims

## 📈 Evaluation System (LangSmith-based)

### Metrics Tracked
- Response accuracy
- Token usage
- Latency per agent
- Citation quality
- Format compliance

### Continuous Improvement
- A/B testing via LangSmith prompt versions
- Performance monitoring dashboards
- Automated regression detection

## 🚨 Error Handling & Stability

### Resilience Features
1. **Timeout Management**: 2-minute global timeout with retry logic
2. **Graceful Degradation**: Falls back to alternative data sources
3. **Rate Limiting**: Max 3 concurrent BrightData requests
4. **Error Recovery**: Automatic retry with exponential backoff

### Never Crashes
- All exceptions caught and logged
- User-friendly error messages
- Partial results returned when possible

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- BrightData account with API access
- OpenAI API key
- LangSmith account (required for prompt management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sdr-agent.git
cd sdr-agent
```

2. Create virtual environment with uv:
```bash
uv venv
uv pip install -e .
```

3. Configure environment variables:

Create a `.env` file with:

```dotenv
OPENAI_API_KEY="sk-..."
BRIGHTDATA_API_KEY="..."

# These are the default Bright Data zones, update if you have custom ones
BRIGHTDATA_WEB_UNLOCKER_ZONE="web_unlocker1"
BRIGHTDATA_BROWSER_ZONE="scraping_browser3"
BRIGHTDATA_SERP_ZONE="serp_api1"

# LangSmith is REQUIRED for prompt management
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="sdr-agent"
# LANGSMITH_OWNER="your-username"  # Optional: specify if prompts are under a specific owner
```

### 3. Running the Agent

```bash
# Simple company research
uv run python main.py run "Tell me about Stripe"

# Contact discovery
uv run python main.py run "Find senior PM leaders at Shopify with emails"

# Complex analysis
uv run python main.py run "Compare Zoom vs Teams market share and suggest which is easier to displace"

# With JSON output
uv run python main.py run '{"query": "Research OpenAI", "format": "json"}'
```

## 📝 Example Queries

### Simple (uses gpt-4o)
- "What's Databricks' main product?"
- "Find the CTO of Anthropic"
- "Is Notion a good prospect for DevOps tools?"

### Complex (uses o4-mini)
- "Analyze Stripe's latest funding round with investor details and valuation"
- "Compare market positioning of Asana vs Monday.com for enterprise sales"
- "Find decision makers at Spotify and draft personalized outreach for each"

## 🔧 Development

### Project Structure
```
sdr-agent/
├── agents/          # Individual agent implementations
├── core/            # Core workflow and configuration
├── tools/           # External integrations (BrightData, search)
├── utils/           # Helper utilities
└── main.py          # CLI entry point
```

### Adding New Agents
1. Create agent class in `agents/`
2. Implement required interface methods
3. Register in `core/graph.py`
4. Add prompts to LangSmith

### Testing
```bash
# Run specific test query
uv run python main.py run "Your test query" --verbose

# Check LangSmith traces
# Visit: https://smith.langchain.com/projects/your-project
```

## 🚀 Future Enhancements

1. **Perplexity-style Deep Search**: Integration with BrightData's new deep_search endpoint
2. **Continuous Learning**: Automated prompt optimization based on user feedback
3. **Additional Data Sources**: CRM integrations, social media signals
4. **Streaming Responses**: Real-time output as agents complete

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 💬 Support

For issues and questions:
- Open a GitHub issue
- Check LangSmith traces for debugging
- Review logs for detailed error information 