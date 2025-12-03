import logging
import asyncio
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with learning style and preferences"""
    user_id: int
    name: Optional[str] = None
    learning_style: str = "balanced"  # visual, auditory, kinesthetic, balanced
    expertise_level: str = "intermediate"  # beginner, intermediate, advanced
    preferred_topics: List[str] = field(default_factory=list)
    language_preference: str = "english"
    response_style: str = "balanced"  # concise, detailed, technical, casual
    interests: Dict[str, float] = field(default_factory=dict)  # topic -> engagement_score
    interaction_count: int = 0
    last_updated: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "learning_style": self.learning_style,
            "expertise_level": self.expertise_level,
            "preferred_topics": self.preferred_topics,
            "language_preference": self.language_preference,
            "response_style": self.response_style,
            "interests": self.interests,
            "interaction_count": self.interaction_count,
            "last_updated": self.last_updated,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data.get("user_id"),
            name=data.get("name"),
            learning_style=data.get("learning_style", "balanced"),
            expertise_level=data.get("expertise_level", "intermediate"),
            preferred_topics=data.get("preferred_topics", []),
            language_preference=data.get("language_preference", "english"),
            response_style=data.get("response_style", "balanced"),
            interests=data.get("interests", {}),
            interaction_count=data.get("interaction_count", 0),
            last_updated=data.get("last_updated", time.time()),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ConversationMemory:
    """Structured conversation memory with semantic embeddings"""
    message_id: str
    user_id: int
    content: str
    role: str  # user, assistant
    timestamp: float
    topic: Optional[str] = None
    importance_score: float = 0.5
    context_relevance: float = 0.0
    decay_factor: float = 1.0  # Decreases with time
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp,
            "topic": self.topic,
            "importance_score": self.importance_score,
            "context_relevance": self.context_relevance,
            "decay_factor": self.decay_factor,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class PersonalizedRAGSystem:
    """
    Personalized Retrieval-Augmented Generation System
    Enhances memory recall with user profiling and semantic search
    """

    def __init__(self, memory_manager, persistence_manager, db=None):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager
        self.persistence_manager = persistence_manager
        self.db = db

        # RAG Configuration
        self.user_profiles: Dict[int, UserProfile] = {}
        self.conversation_memories: Dict[int, List[ConversationMemory]] = {}
        self.topic_cache: Dict[int, Dict[str, float]] = {}  # user_id -> topic scores

        # Memory decay configuration
        self.recent_threshold = 7 * 24 * 3600  # 7 days
        self.active_threshold = 30 * 24 * 3600  # 30 days
        self.archive_threshold = 90 * 24 * 3600  # 90 days

        # Personalization factors
        self.learning_style_boost = {
            "visual": {"image": 1.3, "diagram": 1.3, "chart": 1.2},
            "auditory": {"voice": 1.3, "explanation": 1.2},
            "kinesthetic": {"example": 1.3, "tutorial": 1.2, "code": 1.1},
            "balanced": {},
        }

        self.logger.info("PersonalizedRAGSystem initialized")

    async def initialize_user_profile(
        self, user_id: int, initial_data: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """Initialize or load user profile"""
        try:
            # Try to load from database
            if self.db is not None:
                profile_data = await self._load_profile_from_db(user_id)
                if profile_data:
                    profile = UserProfile.from_dict(profile_data)
                    self.user_profiles[user_id] = profile
                    self.logger.info(f"Loaded existing profile for user {user_id}")
                    return profile

            # Create new profile
            profile = UserProfile(user_id=user_id)
            if initial_data:
                profile.name = initial_data.get("name")
                profile.language_preference = initial_data.get(
                    "language_preference", "english"
                )
                profile.learning_style = initial_data.get("learning_style", "balanced")

            self.user_profiles[user_id] = profile
            await self._save_profile_to_db(user_id, profile)
            self.logger.info(f"Created new profile for user {user_id}")
            return profile

        except Exception as e:
            self.logger.error(f"Error initializing user profile: {e}")
            return UserProfile(user_id=user_id)

    async def update_user_profile(
        self, user_id: int, updates: Dict[str, Any]
    ) -> UserProfile:
        """Update user profile with new information"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = await self.initialize_user_profile(user_id)

        # Update profile fields
        if "learning_style" in updates:
            profile.learning_style = updates["learning_style"]
        if "expertise_level" in updates:
            profile.expertise_level = updates["expertise_level"]
        if "response_style" in updates:
            profile.response_style = updates["response_style"]
        if "name" in updates:
            profile.name = updates["name"]
        if "language_preference" in updates:
            profile.language_preference = updates["language_preference"]

        # Add interests if topics mentioned
        if "interests" in updates:
            profile.interests.update(updates["interests"])

        profile.interaction_count += 1
        profile.last_updated = time.time()

        self.user_profiles[user_id] = profile
        await self._save_profile_to_db(user_id, profile)
        return profile

    async def add_memory(
        self,
        user_id: int,
        content: str,
        role: str,
        topic: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMemory:
        """Add a memory to user's personalized memory store"""
        try:
            if user_id not in self.conversation_memories:
                self.conversation_memories[user_id] = []

            # Create memory object
            memory = ConversationMemory(
                message_id=f"mem_{user_id}_{int(time.time() * 1000)}",
                user_id=user_id,
                content=content,
                role=role,
                timestamp=time.time(),
                topic=topic or self._extract_topic(content),
                importance_score=importance,
                tags=tags or [],
                metadata=metadata or {},
            )

            # Store in memory
            self.conversation_memories[user_id].append(memory)

            # Update user interests based on topic
            if memory.topic:
                await self._update_user_interests(user_id, memory.topic)

            # Persist to database
            if self.db is not None:
                await self._save_memory_to_db(user_id, memory)

            self.logger.debug(
                f"Added memory for user {user_id}: {memory.message_id}"
            )
            return memory

        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            raise

    async def retrieve_personalized_context(
        self,
        user_id: int,
        query: str,
        limit: int = 5,
        include_learning_style: bool = True,
        include_expertise_boost: bool = True,
    ) -> List[ConversationMemory]:
        """
        Retrieve memories relevant to query with personalization
        Takes into account user's learning style and expertise level
        """
        try:
            profile = self.user_profiles.get(user_id)
            if not profile:
                profile = await self.initialize_user_profile(user_id)

            memories = self.conversation_memories.get(user_id, [])
            if not memories:
                self.logger.debug(f"No memories found for user {user_id}")
                return []

            # Calculate relevance scores
            scored_memories = []
            current_time = time.time()

            for memory in memories:
                # Semantic similarity score
                semantic_score = self._calculate_semantic_similarity(
                    query, memory.content
                )

                # Recency score (decay over time)
                age_seconds = current_time - memory.timestamp
                decay = self._calculate_memory_decay(age_seconds)

                # Topic relevance
                topic_score = self._calculate_topic_relevance(
                    memory.topic, profile.interests
                )

                # Learning style boost
                learning_boost = 1.0
                if include_learning_style and profile.learning_style != "balanced":
                    learning_boost = self._calculate_learning_style_boost(
                        memory.content, profile.learning_style
                    )

                # Expertise level adjustment
                expertise_boost = 1.0
                if include_expertise_boost:
                    expertise_boost = self._calculate_expertise_boost(
                        memory.content, profile.expertise_level
                    )

                # Combined score
                combined_score = (
                    (semantic_score * 0.4)
                    + (topic_score * 0.2)
                    + (memory.importance_score * 0.2)
                    + (decay * 0.2)
                ) * learning_boost * expertise_boost

                memory.context_relevance = combined_score
                scored_memories.append((memory, combined_score))

            # Sort by score and return top memories
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            result = [mem for mem, score in scored_memories[:limit]]

            self.logger.debug(
                f"Retrieved {len(result)} personalized memories for user {user_id}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving personalized context: {e}")
            return []

    async def get_learning_path_recommendations(
        self, user_id: int, current_topic: str
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized learning path recommendations based on user profile
        and interaction history
        """
        try:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return []

            memories = self.conversation_memories.get(user_id, [])
            if not memories:
                return []

            # Analyze topic progression
            topics_covered = set()
            topic_sequence = []

            for memory in sorted(memories, key=lambda m: m.timestamp):
                if memory.topic and memory.role == "assistant":
                    topics_covered.add(memory.topic)
                    topic_sequence.append(memory.topic)

            # Get related topics not yet covered
            all_topics = set(profile.interests.keys())
            uncovered_topics = all_topics - topics_covered

            # Generate recommendations
            recommendations = []
            for topic in uncovered_topics:
                interest_score = profile.interests.get(topic, 0)
                recommendation = {
                    "topic": topic,
                    "interest_score": interest_score,
                    "suggested_depth": self._suggest_depth_level(
                        topic, profile.expertise_level
                    ),
                    "learning_resources": self._suggest_learning_resources(
                        topic, profile.learning_style
                    ),
                    "estimated_time": self._estimate_learning_time(
                        topic, profile.expertise_level
                    ),
                }
                recommendations.append(recommendation)

            # Sort by interest score
            recommendations.sort(
                key=lambda x: x["interest_score"], reverse=True
            )
            return recommendations[:5]

        except Exception as e:
            self.logger.error(f"Error generating learning path: {e}")
            return []

    async def consolidate_memories(
        self, user_id: int, days: int = 30
    ) -> Dict[str, Any]:
        """
        Consolidate old memories into summaries to save storage
        Returns consolidated data and cleanup statistics
        """
        try:
            memories = self.conversation_memories.get(user_id, [])
            if not memories:
                return {"consolidated": 0, "archived": 0}

            current_time = time.time()
            consolidation_cutoff = current_time - (days * 24 * 3600)

            old_memories = [
                m for m in memories if m.timestamp < consolidation_cutoff
            ]
            active_memories = [
                m for m in memories if m.timestamp >= consolidation_cutoff
            ]

            # Create summary of old memories
            if old_memories:
                summary = self._create_memory_summary(old_memories)

                # Store consolidated summary
                if self.db is not None:
                    await self._save_consolidated_summary(user_id, summary)

                # Update memory store
                self.conversation_memories[user_id] = active_memories
                self.logger.info(
                    f"Consolidated {len(old_memories)} memories for user {user_id}"
                )

                return {
                    "consolidated": len(old_memories),
                    "active_remaining": len(active_memories),
                    "summary": summary,
                }

            return {"consolidated": 0, "archived": 0}

        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            return {"error": str(e)}

    async def search_by_topic(
        self, user_id: int, topic: str, limit: int = 10
    ) -> List[ConversationMemory]:
        """Search memories by specific topic"""
        memories = self.conversation_memories.get(user_id, [])
        if not memories:
            return []

        matching = [m for m in memories if m.topic == topic]
        matching.sort(key=lambda m: m.timestamp, reverse=True)
        return matching[:limit]

    async def export_user_learning_profile(
        self, user_id: int
    ) -> Dict[str, Any]:
        """Export comprehensive learning profile for analysis"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = await self.initialize_user_profile(user_id)

        memories = self.conversation_memories.get(user_id, [])

        # Calculate statistics
        total_interactions = len(memories)
        topics_covered = len(set(m.topic for m in memories if m.topic))
        assistant_turns = len([m for m in memories if m.role == "assistant"])
        user_turns = len([m for m in memories if m.role == "user"])

        # Get top topics
        topic_counts = {}
        for memory in memories:
            if memory.topic:
                topic_counts[memory.topic] = topic_counts.get(memory.topic, 0) + 1

        top_topics = sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "user_id": user_id,
            "profile": profile.to_dict(),
            "statistics": {
                "total_interactions": total_interactions,
                "topics_covered": topics_covered,
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
                "avg_interaction_length": (
                    sum(len(m.content) for m in memories) / total_interactions
                    if total_interactions
                    else 0
                ),
            },
            "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
            "interests": profile.interests,
            "export_timestamp": time.time(),
        }

    # ============= Helper Methods =============

    def _extract_topic(self, content: str) -> Optional[str]:
        """Extract main topic from content"""
        # Simple keyword-based extraction
        keywords = [
            "python",
            "javascript",
            "learning",
            "tutorial",
            "question",
            "help",
            "explain",
            "code",
            "function",
            "data",
            "api",
            "database",
            "design",
            "architecture",
        ]

        content_lower = content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                return keyword

        # Default to first few words
        words = content.split()[:3]
        return " ".join(words) if words else None

    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """Calculate semantic similarity between query and content"""
        try:
            # Simple word overlap similarity
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())

            if not query_words or not content_words:
                return 0.0

            intersection = len(query_words & content_words)
            union = len(query_words | content_words)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            self.logger.debug(f"Error calculating similarity: {e}")
            return 0.0

    def _calculate_memory_decay(self, age_seconds: float) -> float:
        """Calculate decay factor based on memory age"""
        # Recent memories have higher value
        if age_seconds < self.recent_threshold:
            return 1.0
        elif age_seconds < self.active_threshold:
            # Decay from 1.0 to 0.5
            decay = 1.0 - 0.5 * (
                (age_seconds - self.recent_threshold)
                / (self.active_threshold - self.recent_threshold)
            )
            return max(decay, 0.5)
        elif age_seconds < self.archive_threshold:
            # Decay from 0.5 to 0.1
            decay = 0.5 - 0.4 * (
                (age_seconds - self.active_threshold)
                / (self.archive_threshold - self.active_threshold)
            )
            return max(decay, 0.1)
        else:
            return 0.05

    def _calculate_topic_relevance(
        self, topic: Optional[str], interests: Dict[str, float]
    ) -> float:
        """Calculate relevance of topic based on user interests"""
        if not topic:
            return 0.2

        return interests.get(topic, 0.3)

    def _calculate_learning_style_boost(
        self, content: str, learning_style: str
    ) -> float:
        """Apply learning style-specific boost to content"""
        boost_dict = self.learning_style_boost.get(learning_style, {})
        if not boost_dict:
            return 1.0

        content_lower = content.lower()
        max_boost = 1.0

        for marker, boost_factor in boost_dict.items():
            if marker in content_lower:
                max_boost = max(max_boost, boost_factor)

        return max_boost

    def _calculate_expertise_boost(
        self, content: str, expertise_level: str
    ) -> float:
        """Apply expertise level-specific boost to content"""
        # Complex technical content for advanced users
        complex_markers = [
            "algorithm",
            "optimization",
            "architecture",
            "pattern",
            "framework",
        ]
        simple_markers = ["basic", "intro", "simple", "beginner", "fundamental"]

        content_lower = content.lower()

        if expertise_level == "advanced":
            if any(m in content_lower for m in complex_markers):
                return 1.2
            elif any(m in content_lower for m in simple_markers):
                return 0.8

        elif expertise_level == "beginner":
            if any(m in content_lower for m in simple_markers):
                return 1.2
            elif any(m in content_lower for m in complex_markers):
                return 0.6

        return 1.0

    async def _update_user_interests(self, user_id: int, topic: str) -> None:
        """Update user interests based on interaction"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return

        current_score = profile.interests.get(topic, 0.0)
        new_score = min(current_score + 0.1, 1.0)
        profile.interests[topic] = new_score

    def _suggest_depth_level(
        self, topic: str, expertise_level: str
    ) -> str:
        """Suggest depth level for learning topic"""
        level_map = {
            "beginner": "foundational",
            "intermediate": "intermediate",
            "advanced": "deep_dive",
        }
        return level_map.get(expertise_level, "intermediate")

    def _suggest_learning_resources(
        self, topic: str, learning_style: str
    ) -> List[str]:
        """Suggest learning resources based on style"""
        resource_map = {
            "visual": ["diagrams", "videos", "charts"],
            "auditory": ["podcasts", "lectures", "explanations"],
            "kinesthetic": ["tutorials", "hands-on", "examples"],
            "balanced": ["mixed_media", "documentation", "examples"],
        }
        return resource_map.get(learning_style, [])

    def _estimate_learning_time(self, topic: str, expertise_level: str) -> str:
        """Estimate time needed to learn topic"""
        time_map = {
            "beginner": "30-60 minutes",
            "intermediate": "1-2 hours",
            "advanced": "2-4 hours",
        }
        return time_map.get(expertise_level, "1-2 hours")

    def _create_memory_summary(
        self, memories: List[ConversationMemory]
    ) -> Dict[str, Any]:
        """Create summary of consolidated memories"""
        topics = set()
        content_preview = []

        for memory in memories[:5]:  # Take first 5
            if memory.topic:
                topics.add(memory.topic)
            content_preview.append(memory.content[:100])

        return {
            "consolidated_count": len(memories),
            "topics": list(topics),
            "preview": content_preview,
            "consolidated_at": time.time(),
        }

    # ============= Database Methods =============

    async def _load_profile_from_db(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Load user profile from database"""
        try:
            if self.db is None:
                return None

            collection = self.db["user_profiles"]
            profile = await asyncio.to_thread(
                collection.find_one, {"user_id": user_id}
            )
            return profile

        except Exception as e:
            self.logger.error(f"Error loading profile from DB: {e}")
            return None

    async def _save_profile_to_db(
        self, user_id: int, profile: UserProfile
    ) -> None:
        """Save user profile to database"""
        try:
            if self.db is None:
                return

            collection = self.db["user_profiles"]
            await asyncio.to_thread(
                collection.update_one,
                {"user_id": user_id},
                {"$set": profile.to_dict()},
                upsert=True,
            )

        except Exception as e:
            self.logger.error(f"Error saving profile to DB: {e}")

    async def _save_memory_to_db(
        self, user_id: int, memory: ConversationMemory
    ) -> None:
        """Save memory to database"""
        try:
            if self.db is None:
                return

            collection = self.db["personalized_memories"]
            await asyncio.to_thread(
                collection.insert_one, memory.to_dict()
            )

        except Exception as e:
            self.logger.error(f"Error saving memory to DB: {e}")

    async def _save_consolidated_summary(
        self, user_id: int, summary: Dict[str, Any]
    ) -> None:
        """Save consolidated memory summary"""
        try:
            if self.db is None:
                return

            collection = self.db["consolidated_memories"]
            summary["user_id"] = user_id
            await asyncio.to_thread(collection.insert_one, summary)

        except Exception as e:
            self.logger.error(f"Error saving consolidated summary: {e}")
