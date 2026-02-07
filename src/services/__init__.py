"""
Services Package.

This package contains business logic services that orchestrate
data processing, analysis, and reporting.
"""

from .menu_analysis_service import MenuAnalysisService
from .reporting_service import ReportingService
from .inventory_analysis_service import InventoryAnalysisService
from .inventory_visualizations import InventoryVisualizationService
from .chat_service import ChatService
from .llm_service import LLMService, LLMConfig, LLMProvider, LLMError
from .data_context_builder import DataContextBuilder
from .conversation_manager import ConversationManager

__all__ = [
    # Legacy BCG analysis services
    'MenuAnalysisService',
    'ReportingService',
    # Inventory optimization services
    'InventoryAnalysisService',
    'InventoryVisualizationService',
    # Chat / LLM services
    'ChatService',
    'LLMService',
    'LLMConfig',
    'LLMProvider',
    'LLMError',
    'DataContextBuilder',
    'ConversationManager',
]
