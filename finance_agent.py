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
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True, analyst_recommendations=True, stock_fundamentals=True), get_company_symbol],
    show_tool_calls=True, markdown=True, instructions=["Use labels to display data. If you do not know the company symbol, please use this tool, even if it is not a public company"], debug_mode=True
)


finacial_agent.print_response(
    "Summarize and compare analyst recommendations and fundamentals for TSLA and Phidata. Show in tables.", stream=True
)
