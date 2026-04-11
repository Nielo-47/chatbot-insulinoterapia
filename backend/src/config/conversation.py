import os


CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "50"))

# Summarization settings
SUMMARIZE_MAX_MESSAGES = int(os.getenv("SUMMARIZE_MAX_MESSAGES", "20"))