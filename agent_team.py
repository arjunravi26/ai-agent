from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
load_dotenv()


def get_company_symbol(company):
    symbols = {
        "Phidata": "MSFT",
        "Infosys": "INFY",
        "Tesla": "TSLA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
    }
    return symbols.get(company, "Unknown")


finacial_agent = Agent(
    name='Financial Agent',
    role='Get financial data',
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True, analyst_recommendations=True, stock_fundamentals=True), get_company_symbol],
    show_tool_calls=True, markdown=True, instructions=["Use labels to display data. If you do not know the company symbol, please use this tool, even if it is not a public company"],
    debug_mode=True
)

web_agent = Agent(
    name='Web Agent',
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()], instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

agent_team = Agent(
    name='Agent Team',
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finacial_agent],
    instructions=["Always include sources", "Use tables to display data."],
    show_tool_calls=True,
    markdown=True
)
agent_team.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA", stream=True)
