# Integration code to add to TextHandler class
# Add this to the __init__ method of TextHandler

async def _initialize_rag_system(self):
    """Initialize Personalized RAG system for enhanced memory"""
    try:
        from src.services.memory_context.personalized_rag_system import (
            PersonalizedRAGSystem,
        )
        from src.services.memory_context.rag_integration import RAGIntegration

        # Initialize RAG system
        rag_system = PersonalizedRAGSystem(
            memory_manager=self.memory_manager,
            persistence_manager=self.memory_manager.persistence_manager,
            db=self.user_data_manager.db if hasattr(self.user_data_manager, "db") else None,
        )

        self.rag_integration = RAGIntegration(rag_system)
        self.logger.info("Personalized RAG system initialized")
        return True

    except Exception as e:
        self.logger.error(f"Failed to initialize RAG system: {e}")
        return False


# Patch method: Add to _handle_text_conversation method, after getting enhanced_prompt
async def _enhance_prompt_with_rag(
    self, user_id: int, message_text: str, preferred_model: str
) -> str:
    """Enhance prompt with personalized RAG context"""
    try:
        if not hasattr(self, "rag_integration"):
            return message_text

        # Get personalized context
        context = await self.rag_integration.get_context_for_response(
            user_id=user_id, user_query=message_text, limit=5
        )

        if not context.get("personalization_applied"):
            return message_text

        # Build enhanced prompt
        memories_text = ""
        if context.get("relevant_memories"):
            memories_text = "\n\nRELEVANT CONTEXT FROM HISTORY:\n"
            for mem in context["relevant_memories"][:3]:
                memories_text += f"- {mem['content'][:100]}...\n"

        enhanced_prompt = message_text + memories_text

        self.logger.debug(
            f"Enhanced prompt for user {user_id} with RAG context (learner style: {context.get('learning_style')})"
        )

        return enhanced_prompt

    except Exception as e:
        self.logger.debug(f"RAG enhancement failed: {e}")
        return message_text


# Patch method: Add after saving message pair
async def _log_interaction_to_rag(
    self, user_id: int, user_message: str, ai_response: str, model_id: str
) -> None:
    """Log interaction to personalized memory"""
    try:
        if not hasattr(self, "rag_integration"):
            return

        await self.rag_integration.process_user_message(
            user_id=user_id,
            message_content=user_message,
            role="user",
            importance=0.6,
        )

        await self.rag_integration.process_ai_response(
            user_id=user_id,
            response_content=ai_response,
            importance=0.5,
        )

        self.logger.debug(f"Logged interaction to RAG for user {user_id}")

    except Exception as e:
        self.logger.debug(f"Failed to log to RAG: {e}")
