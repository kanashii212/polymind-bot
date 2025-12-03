import logging
from typing import Dict, List, Any, Optional
from .personalized_rag_system import PersonalizedRAGSystem

logger = logging.getLogger(__name__)


class RAGIntegration:
    """
    Integration layer to connect PersonalizedRAG with TextHandler
    Provides simplified API for RAG operations in message processing
    """

    def __init__(self, rag_system: PersonalizedRAGSystem):
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)

    async def process_user_message(
        self,
        user_id: int,
        message_content: str,
        role: str = "user",
        topic: Optional[str] = None,
        importance: float = 0.5,
    ) -> None:
        """
        Process incoming user message and store in personalized memory.
        This should be called after user sends a message.
        """
        try:
            await self.rag_system.add_memory(
                user_id=user_id,
                content=message_content,
                role=role,
                topic=topic,
                importance=importance,
                tags=self._extract_tags(message_content),
            )
            self.logger.debug(f"Processed user message for {user_id}")
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}")

    async def process_ai_response(
        self,
        user_id: int,
        response_content: str,
        topic: Optional[str] = None,
        importance: float = 0.5,
    ) -> None:
        """
        Process AI response and store in personalized memory.
        This should be called after AI generates a response.
        """
        try:
            await self.rag_system.add_memory(
                user_id=user_id,
                content=response_content,
                role="assistant",
                topic=topic,
                importance=importance,
            )
            self.logger.debug(f"Processed AI response for {user_id}")
        except Exception as e:
            self.logger.error(f"Error processing AI response: {e}")

    async def get_context_for_response(
        self,
        user_id: int,
        user_query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Get personalized context before generating AI response.
        Use this when you need to inject user's learning preferences.

        Returns:
            {
                "relevant_memories": [...],
                "user_profile": {...},
                "learning_style": "...",
                "expertise_level": "...",
                "should_include_examples": bool,
                "response_tone": str,
            }
        """
        try:
            # Get personalized context
            memories = await self.rag_system.retrieve_personalized_context(
                user_id=user_id,
                query=user_query,
                limit=limit,
                include_learning_style=True,
                include_expertise_boost=True,
            )

            # Get user profile
            profile = self.rag_system.user_profiles.get(user_id)
            if not profile:
                profile = await self.rag_system.initialize_user_profile(user_id)

            # Build context
            context = {
                "relevant_memories": [
                    {
                        "content": m.content,
                        "role": m.role,
                        "topic": m.topic,
                        "relevance_score": m.context_relevance,
                        "timestamp": m.timestamp,
                    }
                    for m in memories
                ],
                "user_profile": {
                    "name": profile.name,
                    "learning_style": profile.learning_style,
                    "expertise_level": profile.expertise_level,
                    "response_style": profile.response_style,
                    "language_preference": profile.language_preference,
                    "interaction_count": profile.interaction_count,
                },
                "learning_style": profile.learning_style,
                "expertise_level": profile.expertise_level,
                "should_include_examples": profile.learning_style in [
                    "kinesthetic",
                    "balanced",
                ],
                "response_tone": self._get_response_tone(profile.response_style),
                "personalization_applied": True,
            }

            self.logger.debug(f"Retrieved context for {user_id}: {len(memories)} memories")
            return context

        except Exception as e:
            self.logger.error(f"Error getting context for response: {e}")
            return {"error": str(e), "personalization_applied": False}

    async def build_personalized_system_prompt(
        self, user_id: int, base_prompt: str
    ) -> str:
        """
        Enhance base system prompt with user's learning preferences.
        Use this when setting the system message for AI models.
        """
        try:
            profile = self.rag_system.user_profiles.get(user_id)
            if not profile:
                profile = await self.rag_system.initialize_user_profile(user_id)

            personalization = f"""

PERSONALIZATION CONTEXT:
- User Learning Style: {profile.learning_style}
- Expertise Level: {profile.expertise_level}
- Response Style Preference: {profile.response_style}
- Language: {profile.language_preference}

ADAPT YOUR RESPONSES:
"""

            if profile.learning_style == "visual":
                personalization += "- Use diagrams, charts, and visual descriptions\n"
            elif profile.learning_style == "auditory":
                personalization += "- Use clear explanations and spoken-language structure\n"
            elif profile.learning_style == "kinesthetic":
                personalization += (
                    "- Provide hands-on examples, code snippets, and practical exercises\n"
                )

            if profile.expertise_level == "beginner":
                personalization += "- Explain concepts from first principles\n"
                personalization += "- Avoid jargon or define technical terms clearly\n"
            elif profile.expertise_level == "advanced":
                personalization += "- Assume foundational knowledge\n"
                personalization += "- Focus on advanced patterns and optimizations\n"

            if profile.response_style == "concise":
                personalization += "- Keep responses brief and to the point\n"
            elif profile.response_style == "detailed":
                personalization += "- Provide comprehensive explanations with multiple perspectives\n"

            return base_prompt + personalization

        except Exception as e:
            self.logger.error(f"Error building personalized prompt: {e}")
            return base_prompt

    async def get_user_interests(self, user_id: int) -> Dict[str, float]:
        """Get user's tracked interests and engagement scores"""
        profile = self.rag_system.user_profiles.get(user_id)
        if profile:
            return profile.interests
        return {}

    async def get_learning_recommendations(
        self, user_id: int, current_topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get personalized learning recommendations for user.
        """
        try:
            recommendations = (
                await self.rag_system.get_learning_path_recommendations(
                    user_id,
                    current_topic or "general",
                )
            )
            return recommendations
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []

    async def update_user_preferences(
        self, user_id: int, updates: Dict[str, Any]
    ) -> None:
        """
        Update user's learning preferences and profile.
        """
        try:
            await self.rag_system.update_user_profile(user_id, updates)
            self.logger.info(f"Updated preferences for user {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating preferences: {e}")

    async def cleanup_old_memories(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Clean up and consolidate old memories.
        Run periodically to maintain performance.
        """
        try:
            result = await self.rag_system.consolidate_memories(user_id, days)
            self.logger.info(f"Consolidated memories for user {user_id}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            return {"error": str(e)}

    async def export_learning_profile(self, user_id: int) -> Dict[str, Any]:
        """Export comprehensive learning profile"""
        try:
            profile = await self.rag_system.export_user_learning_profile(user_id)
            return profile
        except Exception as e:
            self.logger.error(f"Error exporting profile: {e}")
            return {"error": str(e)}

    # ============= Helper Methods =============

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content"""
        tags = []

        # Detect content type
        if "code" in content.lower() or "```" in content:
            tags.append("code")
        if "?" in content:
            tags.append("question")
        if "error" in content.lower() or "bug" in content.lower():
            tags.append("error_report")
        if "help" in content.lower():
            tags.append("help_request")
        if any(word in content.lower() for word in ["remember", "save", "note"]):
            tags.append("important")

        return tags

    def _get_response_tone(self, style: str) -> str:
        """Map response style to tone"""
        tone_map = {
            "concise": "brief and direct",
            "detailed": "comprehensive and thorough",
            "technical": "technical and precise",
            "casual": "conversational and friendly",
            "balanced": "clear and balanced",
        }
        return tone_map.get(style, "neutral")
