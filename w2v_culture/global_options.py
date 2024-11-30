"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
# Hardware options
N_CORES: int = 8  # max number of CPU cores to use
RAM_CORENLP: str = "32G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 100 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

# Directory locations
os.environ["CORENLP_HOME"] = "/home/zc_research/TAD/w2v_culture/stanford-corenlp-full-2018-10-05/"

# Input data options
INPUT_file = os.path.join('..', 'narrativesBERT','data', 'earnings_calls_20231017.csv')

# Composite key in earnings call data
UNIQUE_KEYS = ['companyid', 'gvkey', 'mostimportantdateutc', 'componentorder', 'transcriptcomponenttypename'] # composite key in earnings call data
SELECTED_COLS = ['companyid', 'gvkey', 'mostimportantdateutc', 'componentorder', 'transcriptcomponenttypename', 'transcriptid', 'speakertypeid', 'componenttext', 'word_count', 'year', 'isdelayed_flag', 'transcriptcomponentid', 'keydevid', 'proid', 'transcriptpersonname']
PROJECT_DIR = os.getcwd()

DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
MODEL_FOLDER = os.path.join(PROJECT_DIR, "models")
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "outputs")

DATA_FOLDER_W2V = os.path.join(PROJECT_DIR, "data")
MODEL_FOLDER_W2V = os.path.join(PROJECT_DIR, "models")
OUTPUT_FOLDER_W2V = os.path.join(PROJECT_DIR, "outputs")

output_fig_folder = os.path.join(PROJECT_DIR, "fig")
data_filename = 'earnings_calls_20231017.csv'
DATE_COLUMN = "transcriptcreationdate_utc"
TOPIC_SCATTER_PLOT = os.path.join(output_fig_folder, "topic_scatter_plot.pdf")
TEXT_COLUMN = "componenttext" # the column in the main earnings call data that contains the earnings transcript
START_ROWS = 10000000 # start row to read from the csv file
NROWS = 20000000 # number of rows to read from the csv file
CHUNK_SIZE = 1000 # number of rows to read at a time
YEAR_FILTER = 2020 # train the model on data from start year to this year
START_YEAR = 2000 # start year of the data
# Batch Size for Bert Topic Model Training in BERTopic_big_data_hpc.py
BATCH_SIZE = 1000
PHRASE_MAX_VOCAB_SIZE = 10000000

# Parsing and analysis options
STOPWORDS: Set[str] = set(
    Path("resources", "StopWords_Generic.txt").read_text().lower().split()
)  # Set of stopwords from https://sraf.nd.edu/textual-analysis/resources/#StopWords
PHRASE_THRESHOLD: int = 10  # threshold of the phraser module (smaller -> more phrases)
PHRASE_MIN_COUNT: int = 5  # min number of times a bigram needs to appear in the corpus to be considered as a phrase
W2V_DIM: int = 300  # dimension of word2vec vectors
W2V_WINDOW: int = 5  # window size in word2vec
W2V_ITER: int = 20  # number of iterations in word2vec
N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
DICT_RESTRICT_VOCAB = 0.2 # change to a fraction number (e.g. 0.2) to restrict the dictionary vocab in the top 20% of most frequent vocab

# Inputs for constructing the expanded dictionary
DIMS: List[str] = [
    "Revenue",
    "Growth",
    "Profit",
    "Cost",
    "Cash",
    "Debt",
    "Equity",
    "Investment",
    "Dividend",
    "Financial Position",
    "Liquidity",
    "Gross Profit Margin",
    "Operating Margin",
    "Free Cash Flow",
    "Return on Equity",
    "Return on Assets",
    "Return on Investment",
    "Productivity",
    "Asset Impairment",
    "Corporate Tax",
    "Short-term Guidance",
    "Full-year Outlook",
    "Long-term Financial Targets",
    "Industry Forecast",
    "Economic Forecast",
    "Market Share",
    "Competitive Landscape",
    "Challenges",
    "Brand Strength",
    "Customer Acquisition",
    "Customer Retention Rates",
    "Net Promoter Score (NPS)",
    "New Product Launches",
    "Product Mix Changes",
    "Service Quality",
    "Research and Development",
    "Innovation Pipeline",
    "Product Roadmap",
    "Cost-cutting Initiatives",
    "Operational Improvements",
    "Productivity Metrics",
    "Capacity Utilization",
    "Supply Chain Efficiency",
    "Inventory Turnover",
    "Capital Structure",
    "Share Buyback Plans",
    "Dividend Policy",
    "Capital Expenditure Plans",
    "Working Capital Management",
    "Geographic Expansion",
    "Merger and Acquisition Activities",
    "Market Penetration Strategies",
    "Diversification Efforts",
    "Partnerships and Collaborations",
    "Sales Pipeline",
    "Backlog or Order Book Status",
    "Customer Acquisition Costs",
    "Lifetime Value of Customers",
    "Marketing Effectiveness",
    "Sales Force Productivity",
    "Business Unit Breakdowns",
    "Geographic Segment Performance",
    "Product Category Performance",
    "Customer Segment Analysis",
    "Raw Material Costs",
    "Labor Costs",
    "Overhead Expenses",
    "Cost of Goods Sold (COGS)",
    "Selling, General, and Administrative Expenses (SG&A)",
    "Regulatory Challenges",
    "Litigation Updates",
    "Cybersecurity Measures",
    "Foreign Exchange Impact",
    "Interest Rate Sensitivity",
    "Employee Headcount",
    "Employee Turnover Rate",
    "Talent Acquisition and Retention Strategies",
    "Workforce Diversity and Inclusion",
    "Employee Engagement Metrics",
    "Digital Transformation Initiatives",
    "IT Infrastructure Investments",
    "E-commerce Performance",
    "Data Analytics Capabilities",
    "Artificial Intelligence and Machine Learning Applications",
    "Automation",
    "Environmental Initiatives",
    "Social Responsibility Programs",
    "Governance Practices",
    "Carbon Footprint Reduction Efforts",
    "Sustainable Sourcing",
    "Patent Portfolio",
    "Trademark Developments",
    "Licensing Agreements",
    "IP Litigation",
    "Corporate Innovation",
    "Customer Satisfaction Scores",
    "Churn Rate",
    "Average Revenue per User (ARPU)",
    "Customer Lifetime Value (CLV)",
    "Pricing Power",
    "Discount Policies",
    "Dynamic Pricing Initiatives",
    "Bundle Pricing Strategies",
    "Organizational Changes",
    "Executive Leadership Transitions",
    "Board Composition",
    "Subsidiary Performance",
    "Sector-specific KPIs",
    "Regulatory Compliance Metrics",
    "Industry Benchmarking",
    "Political Risk",
    "Macro Economic Risk",
    "Liquidity Risk",
    "Sovereign Risk",
    "Credit Risk",
    "Operational Risk",
    "Legal Risk",
    "Climate Risk",
    "Cybersecurity Risk",
    "Urgent and Timely Action"
]

SEED_WORDS : Dict[str, List[str]] = {
    "Revenue": ["revenue", "income statement", "sales", "top-line", "total revenue"],
    "Growth": ["growth", "expansion", "increase", "rise", "escalation"],
    "Profit": ["profit", "net income", "bottom-line", "earnings", "net profit"],
    "Cost": ["cost", "expenses", "expenditure", "overhead", "costs"],
    "Cash": ["cash", "cash flow", "liquidity", "cash position", "cash balance"],
    "Debt": ["debt", "liabilities", "borrowing", "indebtedness", "debt burden"],
    "Equity": ["equity", "shareholders", "stockholders", "ownership", "equity holders"],
    "Investment": ["investment", "investing", "capital expenditure", "capex", "investment spending"],
    "Dividend": ["dividend", "dividend payment", "dividend yield", "dividend payout", "dividend policy"],
    "Financial Position": ["financial position", "balance sheet", "financial health", "financial stability", "financial standing"],
    "Liquidity": ["liquidity", "liquid assets", "current assets", "quick ratio", "current ratio"],
    "Gross Profit Margin": ["gross margin", "profit ratio", "markup percentage", "gross profit rate", "sales margin"],
    "Operating Margin": ["operating profit margin", "EBIT margin", "operating income margin", "profit margin", "operational efficiency"],
    "Free Cash Flow": ["cash balance", "cash burn rate", "cash conversion cycle", "cash flow", "cash generation", "cash position"],
    "Return on Equity": ["return on equity", "equity returns", "shareholder return", "net income to equity", "equity performance", "profitability ratio"],
    "Return on Assets": ["return on assets", "asset returns", "asset performance", "net income to assets", "asset profitability"],
    "Return on Investment": ["return on investment", "investment returns", "investment performance", "net income to investment", "investment profitability"],
    "Productivity": ["automation", "capacity utilization", "cost cutting", "cost efficiency", "cost reduction", "cost saving", "digital transformation", "efficiency", "labor cost", "labor efficiency", "labor layoff", "labor productivity", "laid off", "lay off"],
    "Asset Impairment": ["allowance", "write-off", "impairment charge", "asset impairment", "goodwill impairment"],
    "Corporate Tax": ["corporate tax", "effective tax rate", "tax liabilities", "tax planning", "tax credits", "deferred taxes"],
    "Short-term Guidance": ["short-term forecast", "upcoming quarter outlook", "near-term projections", "quarterly expectations", "forward guidance"],
    "Full-year Outlook": ["full-year outlook", "annual forecast", "yearly projection", "long-term guidance", "fiscal year outlook", "12-month projection"],
    "Long-term Financial Targets": ["long-term", "multi-year goals", "strategic financial objectives", "extended financial outlook", "long-range targets", "future financial aims", "strategic plan", "market leadership", "sustainable growth"],
    "Industry Forecast": ["industry forecast", "sector outlook", "market projections", "industry trends", "sector expectations"],
    "Economic Forecast": ["economic forecast", "macroeconomic outlook", "economic projections", "economic trends", "economic expectations", "CPI", "inflation", "GDP"],
    "Market Share": ["market share", "market dominance", "market leadership", "market position", "business footprint"],
    "Competitive Landscape": ["competitive landscape", "competitive risk", "competitive environment", "industry rivalry", "market competition"],
    "Challenges": ["challenge", "adjustment period", "bear the burden", "difficult environment", "headwind", "lack of visibility", "materially impacted"],
    "Brand Strength": ["brand strength", "brand power", "brand health", "brand recognition", "brand equity", "brand value"],
    "Customer Acquisition": ["new customer growth", "client onboarding", "customer wins", "new business generation", "expanding customer base"],
    "Customer Retention Rates": ["client loyalty", "churn rate", "customer stickiness", "repeat business", "customer longevity"],
    "Net Promoter Score (NPS)": ["customer satisfaction index", "loyalty metric", "referral likelihood", "customer advocacy", "satisfaction score"],
    "New Product Launches": ["product releases", "new offerings", "product introductions", "market debuts", "new solutions"],
    "Product Mix Changes": ["product portfolio shifts", "offering diversification", "product line adjustments", "sales mix"],
    "Service Quality": ["service performance indicators", "quality assurance metrics", "service level achievements", "customer experience scores"],
    "Research and Development": ["R&D spending", "innovation funding", "product development costs", "research expenditure", "technology investments"],
    "Innovation Pipeline": ["future products", "development roadmap", "upcoming innovations", "product incubation"],
    "Product Roadmap": ["development timeline", "product strategy", "future releases", "product evolution plan"],
    "Cost-cutting Initiatives": ["cost reduction", "budget trimming", "savings measures", "expense management"],
    "Operational Improvements": ["process enhancements", "efficiency gains", "operational streamlining", "productivity boosts"],
    "Productivity Metrics": ["efficiency measures", "output indicators", "performance ratios", "productivity KPIs"],
    "Capacity Utilization": ["resource usage", "production capacity", "facility utilization", "asset efficiency"],
    "Supply Chain Efficiency": ["logistics performance", "supply network optimization", "distribution efficiency"],
    "Inventory Turnover": ["stock rotation", "inventory efficiency", "goods turnover rate", "inventory churn"],
    "Capital Structure": ["borrowings", "financial leverage", "liabilities", "debt capacity", "capital structure optimization"],
    "Share Buyback Plans": ["stock repurchase program", "share repurchases", "buyback initiative", "equity reduction"],
    "Dividend Policy": ["payout policy", "shareholder distributions", "income distribution plan"],
    "Capital Expenditure Plans": ["Capex projections", "investment plans", "infrastructure spending"],
    "Working Capital Management": ["cash flow management", "liquidity management", "short-term asset management"],
    "Geographic Expansion": ["market entry", "territorial growth", "global reach expansion"],
    "Merger and Acquisition Activities": ["M&A strategy", "corporate takeovers", "business combinations", "acquisition plans"],
    "Market Penetration Strategies": ["market share growth", "sales penetration tactics", "customer base expansion"],
    "Diversification Efforts": ["business expansion", "new venture development", "portfolio diversification"],
    "Partnerships and Collaborations": ["strategic alliances", "joint ventures", "business partnerships"],
    "Sales Pipeline": ["sales funnel", "prospect pipeline", "deal flow", "sales forecast"],
    "Backlog or Order Book Status": ["unfilled orders", "work in progress", "order queue"],
    "Customer Acquisition Costs": ["CAC", "cost per customer", "marketing efficiency"],
    "Lifetime Value of Customers": ["LTV", "long-term customer value", "customer profitability"],
    "Marketing Effectiveness": ["ROI on marketing", "campaign performance", "advertising effectiveness"],
    "Sales Force Productivity": ["sales efficiency", "rep performance", "revenue per salesperson"],
    "Business Unit Breakdowns": ["divisional performance", "segment analysis", "unit-level results"],
    "Geographic Segment Performance": ["regional results", "country-specific performance"],
    "Product Category Performance": ["product line results", "category-wise analysis"],
    "Customer Segment Analysis": ["client group performance", "target market results"],
    "Raw Material Costs": ["input costs", "material expenses", "commodity prices"],
    "Labor Costs": ["workforce expenses", "payroll expenses"],
    "Overhead Expenses": ["indirect costs", "fixed costs", "operating expenses"],
    "Cost of Goods Sold (COGS)": ["production costs", "manufacturing expenses", "cost of sales"],
    "Selling, General, and Administrative Expenses (SG&A)": ["operating expenses", "non-production costs"],
    "Regulatory Challenges": ["compliance issues", "legal hurdles", "regulatory environment"],
    "Litigation Updates": ["legal proceedings", "lawsuit status", "legal dispute updates"],
    "Cybersecurity Measures": ["data protection", "information security", "cyber defense"],
    "Foreign Exchange Impact": ["currency effects", "forex exposure", "exchange rate influence"],
    "Interest Rate Sensitivity": ["rate exposure", "interest risk", "borrowing cost sensitivity"],
    "Employee Headcount": ["workforce size", "staff numbers", "employee strength"],
    "Employee Turnover Rate": ["staff attrition", "workforce stability", "employee departures"],
    "Talent Acquisition and Retention Strategies": ["hiring initiatives", "employee retention programs"],
    "Workforce Diversity and Inclusion": ["diversity metrics", "inclusivity efforts"],
    "Employee Engagement Metrics": ["staff satisfaction", "workforce morale", "team engagement"],
    "Digital Transformation Initiatives": ["digitalization efforts", "tech modernization", "digital evolution"],
    "IT Infrastructure Investments": ["tech spending", "system upgrades", "computing resources"],
    "E-commerce Performance": ["online sales", "digital revenue", "web store results"],
    "Data Analytics Capabilities": ["business intelligence", "data-driven insights", "predictive modeling"],
    "Artificial Intelligence and Machine Learning Applications": ["AI integration", "ML implementation", "cognitive computing"],
    "Automation": ["automation", "robotics", "RPA", "process automation"],
    "Environmental Initiatives": ["eco-friendly programs", "green initiatives", "environmental stewardship"],
    "Social Responsibility Programs": ["community initiatives", "social impact", "corporate citizenship"],
    "Governance Practices": ["corporate governance", "board practices", "management oversight"],
    "Carbon Footprint Reduction Efforts": ["emissions reduction", "climate impact mitigation"],
    "Sustainable Sourcing": ["ethical procurement", "responsible sourcing"],
    "Patent Portfolio": ["IP assets", "patent holdings", "proprietary technology"],
    "Trademark Developments": ["brand protection", "intellectual property rights"],
    "Licensing Agreements": ["IP licensing", "technology transfer", "trademark licensing"],
    "IP Litigation": ["patent disputes", "trademark infringement"],
    "Corporate Innovation": ["innovation", "R&D", "breakthrough technologies"],
    "Customer Satisfaction Scores": ["client happiness index", "satisfaction ratings"],
    "Churn Rate": ["customer attrition", "client loss rate"],
    "Average Revenue per User (ARPU)": ["per-customer revenue", "user monetization"],
    "Customer Lifetime Value (CLV)": ["lifetime customer worth", "customer profitability"],
    "Pricing Power": ["price elasticity", "pricing leverage"],
    "Discount Policies": ["price reduction strategies", "promotional pricing"],
    "Dynamic Pricing Initiatives": ["real-time pricing", "adaptive pricing"],
    "Bundle Pricing Strategies": ["package deals", "product bundling"],
    "Organizational Changes": ["structural shifts", "corporate reorganization"],
    "Executive Leadership Transitions": ["C-suite changes", "management shuffle"],
    "Board Composition": ["director lineup", "board structure"],
    "Subsidiary Performance": ["division results", "affiliate performance"],
    "Sector-specific KPIs": ["industry benchmarks", "sector performance indicators"],
    "Regulatory Compliance Metrics": ["compliance scores", "regulatory adherence measures"],
    "Industry Benchmarking": ["peer comparison", "industry standards comparison"],
    "Political Risk": ["political risk", "uncertainty", "volatility"],
    "Macro Economic Risk": ["macro trends", "economic influences", "market conditions"],
    "Liquidity Risk": ["liquidity risk", "financial tightness", "solvency"],
    "Sovereign Risk": ["sovereign change", "sovereign credit rating"],
    "Credit Risk": ["credit risk", "credit uncertainty", "credit exposure"],
    "Operational Risk": ["brand reputation", "business continuity", "incident management"],
    "Legal Risk": ["legal risk", "legal uncertainty", "compliance"],
    "Climate Risk": ["climate disaster", "environmental hazard"],
    "Cybersecurity Risk": ["cybersecurity risk", "cyberattack", "cybercrime"],
    "Urgent and Timely Action": ["urgent", "immediate", "quick", "expedite"]
}