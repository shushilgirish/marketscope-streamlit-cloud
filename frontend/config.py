"""
Configuration module for MarketScope
"""
import os
import importlib
from typing import Dict, Any, List, Type
from dotenv import load_dotenv
load_dotenv(override=True)
class Config:
    """Configuration settings for MarketScope"""
    
    # AWS Credentials
    AWS_SERVER_PUBLIC_KEY = os.getenv("AWS_SERVER_PUBLIC_KEY")
    AWS_SERVER_SECRET_KEY = os.getenv("AWS_SERVER_SECRET_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
    
    # S3 Configuration
    BUCKET_NAME = os.getenv("BUCKET_NAME", "finalproject-product")
    S3_CHUNKS_PATH = os.getenv("S3_CHUNKS_PATH", "book-content/chunks/")
    S3_CHUNKS_FILE = os.getenv("S3_CHUNKS_FILE", "S3D7W4_Marketing_Management_chunks.json")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Replace with your actual key
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    PINECONE_INDEX_NAME_book = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # Add this line for compatibility
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Snowflake Configuration
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
    SNOWFLAKE_DATABASE = "HEALTHCARE_INDUSTRY_CUSTOMER_DATA"
    SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
    SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")  # Added missing role config
    SNOWFLAKE_MCP_PORT = 8004
    
    # Model Configuration
    DEFAULT_MODEL = "gpt-4"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    AVAILABLE_MODELS = {
        "gpt-4o": {"name": "GPT-4 Optimized", "provider": "openai"},
        "claude-3-haiku-20240307": {"name": "Claude 3 Haiku", "provider": "anthropic"},
        "gemini-pro": {"name": "Gemini Pro", "provider": "google"},
        "deepseek-chat": {"name": "DeepSeek Chat", "provider": "deepseek"},
        "grok-1": {"name": "Grok-1", "provider": "grok"}
    }
    
    # MCP Server configuration
    MCP_PORT = 8000  # Main unified MCP server
    API_PORT = 8001  # API server port
    SALES_MCP_PORT = 8002  # Sales analytics MCP server
    MARKET_MCP_PORT = 8003  # Market analysis MCP server
    
    # Server base URLs - use environment variables if available
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    SALES_MCP_URL = os.getenv("SALES_MCP_URL", "http://localhost:8002")
    MARKET_MCP_URL = os.getenv("MARKET_MCP_URL", "http://localhost:8003")
    SNOWFLAKE_MCP_URL = os.getenv("SNOWFLAKE_MCP_URL", "http://localhost:8004")
    
    # Service URLs mapping
    SERVICE_URLS = {
        "marketscope": MCP_SERVER_URL,
        "sales": SALES_MCP_URL,
        "market": MARKET_MCP_URL,
        "snowflake": SNOWFLAKE_MCP_URL
    }
    
    # API endpoints
    API_BASE = "/api/v1"
    MCP_BASE = "/mcp"
    TOOLS_ENDPOINT = "/tools"
    HEALTH_ENDPOINT = "/health"
    
    # Endpoint paths
    def get_tools_path(self):
        """Get the tools endpoint path with fallback"""
        return f"{self.MCP_BASE}{self.TOOLS_ENDPOINT}", self.TOOLS_ENDPOINT
    
    def get_health_path(self):
        """Get the health check endpoint path"""
        return f"{self.HEALTH_ENDPOINT}"
    
    # Tool Configuration
    ENABLED_TOOLS = [
        "pinecone_search",
        "fetch_s3_chunk",
        "get_chunks_metadata",
        "get_all_retrieved_chunks",
        "analyze_market_segment",
        "generate_segment_strategy"
    ]
    
    # Agent Configurations
    AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
        "marketing": {
            "agent_class": "agents.marketing_management_book.agent.MarketingManagementAgent",
            "tool_modules": ["agents.marketing_management_book.marketing_tools"],
            "description": "Marketing Management Agent"
        }
    }
    
    # Segment Configuration - Moved from segment_servers/config.py
    SEGMENT_CONFIG = {
        "Skin Care Segment": {"port": 8014, "namespace": "skincare"},  # Changed from 8003 to 8014
        "Healthcare - Diagnostic": {"port": 8015, "namespace": "diagnostics"},
        "Pharmaceutical": {"port": 8005, "namespace": "otc-pharmaceutical"},
        "Supplements": {"port": 8006, "namespace": "supplements"},
        "Segment Analysis": {"port": 8007, "namespace": "segment-analysis"},
        "Segment Study Reports": {"port": 8008, "namespace": "segment-study-reports"},
        "Wearables": {"port": 8009, "namespace": "wearables"}
    }

    # MCP server names - will be used in the unified agent
    MCP_SERVER_NAMES = {
        "Skin Care Segment": "skincare_mcp",
        "Healthcare - Diagnostic": "diagnostics_mcp",
        "Pharmaceutical": "pharmaceutical_mcp",
        "Diagnostic Segment": "diagnostic_mcp_server",
        "Supplement Segment": "supplement_mcp_server",
        "Otc Pharmaceutical Segment": "otc_pharma_mcp_server",
        "Fitness Wearable Segment": "fitness_wearable_mcp_server"
    }

    # Categories for tools to be used in the unified agent
    TOOL_CATEGORIES = {
        "Diagnostic Segment": "diagnostic",
        "Supplement Segment": "supplement",
        "Otc Pharmaceutical Segment": "otc_pharma",
        "Fitness Wearable Segment": "fitness", 
        "Skin Care Segment": "skincare"
    }

    # Store common tools to be registered by each MCP server
    COMMON_TOOL_NAMES = [
        "upload_to_snowflake",
        "get_product_list",
        "analyze_product_trends",
        "create_sales_visualization",
        "generate_sales_summary"
    ]
    
    @classmethod
    def get_agent_class(cls, agent_name: str) -> Type:
        """Get the agent class from the fully qualified class name."""
        try:
            module_path, class_name = cls.AGENT_CONFIGS[agent_name]["agent_class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (KeyError, ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Could not load agent class for {agent_name}: {str(e)}")

    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate that required configuration is present
        
        Returns:
            List of missing configuration variables
        """
        required_vars = [
            "AWS_SERVER_PUBLIC_KEY", 
            "AWS_SERVER_SECRET_KEY",
            "PINECONE_API_KEY",
            "OPENAI_API_KEY"
        ]
        
        missing = [var for var in required_vars if not getattr(cls, var)]
        
        return missing

    @classmethod
    def get_model_config(cls, model_name: str = None) -> Dict[str, Any]:
        """Get the configuration for a specific model"""
        model_name = model_name or cls.DEFAULT_MODEL
        return cls.AVAILABLE_MODELS.get(model_name, {"name": model_name, "provider": "unknown"})

    @classmethod
    def get_pinecone_api_key(cls):
        """Get Pinecone API key from multiple possible sources"""
        # Check environment variable first
        import os
        env_key = os.environ.get("PINECONE_API_KEY")
        if env_key:
            return env_key
        
        # Then check config value
        if hasattr(cls, "PINECONE_API_KEY") and cls.PINECONE_API_KEY:
            return cls.PINECONE_API_KEY
        
        # Try to load from local secrets file as fallback
        try:
            secrets_path = os.path.join(os.path.dirname(__file__), "secrets.json")
            if os.path.exists(secrets_path):
                import json
                with open(secrets_path, "r") as f:
                    secrets = json.load(f)
                    if "pinecone_api_key" in secrets:
                        return secrets["pinecone_api_key"]
        except Exception as e:
            print(f"Error reading secrets: {e}")
        
        # No valid key found
        return None

