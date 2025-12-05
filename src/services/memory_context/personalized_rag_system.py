import logging
import asyncio
import os
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field

# Environment variable to disable LangChain on resource-constrained hosting
# Set ENABLE_LANGCHAIN_RAG=false for free tier hosting (512MB RAM, 2GB disk)
ENABLE_LANGCHAIN_RAG = os.getenv("ENABLE_LANGCHAIN_RAG", "true").lower() in ("true", "1", "yes")

# LangChain imports for enhanced RAG (skip if disabled or low resources)
LANGCHAIN_AVAILABLE = False
if ENABLE_LANGCHAIN_RAG:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

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
    embedding: Optional[List[float]] = None  # Vector embedding for semantic search
    chunk_id: Optional[str] = None  # ID for chunked content

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
            "embedding": self.embedding,
            "chunk_id": self.chunk_id,
        }


class PersonalizedRAGSystem:
    """
    Personalized Retrieval-Augmented Generation System
    Enhances memory recall with user profiling and semantic search
    
    Features:
    - LangChain integration for advanced RAG capabilities
    - Sentence Transformer embeddings for semantic similarity
    - FAISS vector store for efficient similarity search
    - Hierarchical memory with short-term, medium-term, and long-term tiers
    - Intelligent text chunking for long documents
    - Memory consolidation and summarization
    """

    def __init__(self, memory_manager, persistence_manager, db=None):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager
        self.persistence_manager = persistence_manager
        self.db = db

        # LangChain components for enhanced RAG
        self.embeddings = None
        self.vector_stores: Dict[int, Any] = {}  # user_id -> FAISS vector store
        self.text_splitter = None
        self.langchain_enabled = False
        
        # Initialize LangChain components
        self._initialize_langchain()

        # Long-context configuration
        self.max_context_tokens = 128000  # Support for 128K context models
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        self.max_retrieval_results = 20  # Maximum results from vector search
        self.context_window_size = 10  # Recent messages to always include
        
        # Hierarchical memory tiers
        self.memory_tiers = {
            "short_term": 7 * 24 * 3600,    # 7 days - full detail
            "medium_term": 30 * 24 * 3600,  # 30 days - summarized
            "long_term": 90 * 24 * 3600,    # 90 days - key insights only
        }

        # Memory decay configuration (legacy support)
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

        self.logger.info(
            f"PersonalizedRAGSystem initialized (LangChain: {self.langchain_enabled}, "
            f"ENV_ENABLED: {ENABLE_LANGCHAIN_RAG})"
        )

    def _initialize_langchain(self) -> None:
        """
        Initialize LangChain components for enhanced RAG capabilities.
        
        Resource Requirements (when enabled):
        - Disk: ~2.5GB (torch + transformers + sentence-transformers)
        - RAM: ~400-600MB for model loading
        - CPU: Moderate (embedding generation)
        
        For resource-constrained hosting (e.g., free tier with 512MB RAM):
        Set environment variable: ENABLE_LANGCHAIN_RAG=false
        The system will fall back to basic Jaccard similarity search.
        """
        if not ENABLE_LANGCHAIN_RAG:
            self.logger.info(
                "LangChain RAG disabled via ENABLE_LANGCHAIN_RAG=false (resource-saving mode). "
                "Using basic semantic similarity instead."
            )
            return
            
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning(
                "LangChain not available. Install with: uv add langchain langchain-huggingface langchain-community faiss-cpu sentence-transformers"
            )
            return

        try:
            # Initialize sentence transformer embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            
            # Initialize text splitter for long documents
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size if hasattr(self, 'chunk_size') else 1000,
                chunk_overlap=self.chunk_overlap if hasattr(self, 'chunk_overlap') else 200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            
            self.langchain_enabled = True
            self.logger.info("LangChain components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain: {e}")
            self.langchain_enabled = False

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
        profile_data = await self._load_profile_from_db(user_id)
        if profile_data:
            profile = UserProfile.from_dict(profile_data)
        else:
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
            profile_data = await self._load_profile_from_db(user_id)
            if profile_data:
                profile = UserProfile.from_dict(profile_data)
            else:
                profile = await self.initialize_user_profile(user_id)

            memories = await self._load_memories_from_db(user_id)
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
            profile_data = await self._load_profile_from_db(user_id)
            if not profile_data:
                return []
            profile = UserProfile.from_dict(profile_data)

            memories = await self._load_memories_from_db(user_id)
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
            memories = await self._load_memories_from_db(user_id)
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
        memories = await self._load_memories_from_db(user_id)
        if not memories:
            return []

        matching = [m for m in memories if m.topic == topic]
        matching.sort(key=lambda m: m.timestamp, reverse=True)
        return matching[:limit]

    async def export_user_learning_profile(
        self, user_id: int
    ) -> Dict[str, Any]:
        """Export comprehensive learning profile for analysis"""
        profile_data = await self._load_profile_from_db(user_id)
        if profile_data:
            profile = UserProfile.from_dict(profile_data)
        else:
            profile = await self.initialize_user_profile(user_id)

        memories = await self._load_memories_from_db(user_id)

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

    # ============= LangChain Enhanced Methods for Long-Context =============

    async def add_memory_with_embedding(
        self,
        user_id: int,
        content: str,
        role: str,
        topic: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMemory:
        """
        Add a memory with vector embedding for semantic search.
        Handles long content by chunking if necessary.
        """
        try:
            memories_created = []
            
            # Check if content needs chunking (for long documents)
            if self.langchain_enabled and len(content) > self.chunk_size:
                chunks = self._chunk_content(content)
                self.logger.debug(f"Content chunked into {len(chunks)} parts for user {user_id}")
                
                for i, chunk in enumerate(chunks):
                    chunk_memory = await self._create_memory_with_embedding(
                        user_id=user_id,
                        content=chunk,
                        role=role,
                        topic=topic,
                        importance=importance,
                        tags=tags,
                        metadata={**(metadata or {}), "chunk_index": i, "total_chunks": len(chunks)},
                        chunk_id=f"chunk_{i}",
                    )
                    memories_created.append(chunk_memory)
                
                # Return the first chunk as the primary memory
                return memories_created[0] if memories_created else await self.add_memory(
                    user_id, content, role, topic, importance, tags, metadata
                )
            else:
                return await self._create_memory_with_embedding(
                    user_id, content, role, topic, importance, tags, metadata
                )

        except Exception as e:
            self.logger.error(f"Error adding memory with embedding: {e}")
            # Fallback to basic memory creation
            return await self.add_memory(user_id, content, role, topic, importance, tags, metadata)

    async def _create_memory_with_embedding(
        self,
        user_id: int,
        content: str,
        role: str,
        topic: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None,
    ) -> ConversationMemory:
        """Create a single memory with embedding"""
        embedding = None
        if self.langchain_enabled and self.embeddings:
            try:
                embedding = await asyncio.to_thread(
                    self.embeddings.embed_query, content
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding: {e}")

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
            embedding=embedding,
            chunk_id=chunk_id,
        )

        # Update user interests based on topic
        if memory.topic:
            await self._update_user_interests(user_id, memory.topic)

        # Persist to database
        if self.db is not None:
            await self._save_memory_to_db(user_id, memory)

        # Update vector store for fast retrieval
        if self.langchain_enabled and embedding:
            await self._update_user_vector_store(user_id, memory)

        self.logger.debug(f"Added memory with embedding for user {user_id}: {memory.message_id}")
        return memory

    def _chunk_content(self, content: str) -> List[str]:
        """Split long content into chunks using LangChain text splitter"""
        if not self.langchain_enabled or not self.text_splitter:
            # Fallback to simple chunking
            chunks = []
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunks.append(content[i:i + self.chunk_size])
            return chunks

        try:
            # Use LangChain's RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_text(content)
            return chunks
        except Exception as e:
            self.logger.warning(f"LangChain chunking failed, using fallback: {e}")
            chunks = []
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunks.append(content[i:i + self.chunk_size])
            return chunks

    async def _update_user_vector_store(
        self, user_id: int, memory: ConversationMemory
    ) -> None:
        """Update user's FAISS vector store with new memory"""
        if not self.langchain_enabled or not memory.embedding:
            return

        try:
            if user_id not in self.vector_stores:
                # Create new vector store for user
                self.vector_stores[user_id] = {
                    "embeddings": [memory.embedding],
                    "memories": [memory],
                    "ids": [memory.message_id],
                }
            else:
                # Add to existing store
                self.vector_stores[user_id]["embeddings"].append(memory.embedding)
                self.vector_stores[user_id]["memories"].append(memory)
                self.vector_stores[user_id]["ids"].append(memory.message_id)

        except Exception as e:
            self.logger.error(f"Error updating vector store: {e}")

    async def retrieve_with_semantic_search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> List[ConversationMemory]:
        """
        Retrieve memories using semantic similarity search with embeddings.
        Falls back to basic retrieval if LangChain is not available.
        """
        if not self.langchain_enabled or user_id not in self.vector_stores:
            return await self.retrieve_personalized_context(user_id, query, limit)

        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, query
            )

            user_store = self.vector_stores[user_id]
            embeddings = user_store["embeddings"]
            memories = user_store["memories"]

            # Calculate cosine similarities
            similarities = []
            for i, mem_embedding in enumerate(embeddings):
                similarity = self._cosine_similarity(query_embedding, mem_embedding)
                if similarity >= min_similarity:
                    similarities.append((memories[i], similarity))

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            result = [mem for mem, sim in similarities[:limit]]

            self.logger.debug(
                f"Semantic search retrieved {len(result)} memories for user {user_id}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return await self.retrieve_personalized_context(user_id, query, limit)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    async def retrieve_hierarchical_context(
        self,
        user_id: int,
        query: str,
        max_tokens: int = 8000,
        include_recent: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve context using hierarchical memory strategy for long contexts.
        
        Returns:
            - recent_messages: Always include last N messages for continuity
            - semantic_matches: Relevant memories from semantic search
            - topic_summaries: Summarized context from older memories
            - total_tokens: Estimated token count
        """
        try:
            result = {
                "recent_messages": [],
                "semantic_matches": [],
                "topic_summaries": [],
                "total_tokens": 0,
            }

            # 1. Get recent messages (always include for conversation continuity)
            memories = await self._load_memories_from_db(user_id)
            if not memories:
                return result

            sorted_memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)
            recent = sorted_memories[:include_recent]
            result["recent_messages"] = recent
            
            # Estimate tokens (rough: 4 chars = 1 token)
            recent_tokens = sum(len(m.content) // 4 for m in recent)
            result["total_tokens"] = recent_tokens

            # 2. Semantic search for relevant context
            remaining_tokens = max_tokens - recent_tokens
            if remaining_tokens > 500 and self.langchain_enabled:
                semantic_results = await self.retrieve_with_semantic_search(
                    user_id, query, limit=self.max_retrieval_results
                )
                
                # Filter out recent messages already included
                recent_ids = {m.message_id for m in recent}
                semantic_filtered = [
                    m for m in semantic_results if m.message_id not in recent_ids
                ]

                # Add semantic matches within token budget
                for memory in semantic_filtered:
                    mem_tokens = len(memory.content) // 4
                    if result["total_tokens"] + mem_tokens <= max_tokens:
                        result["semantic_matches"].append(memory)
                        result["total_tokens"] += mem_tokens
                    else:
                        break

            # 3. Add topic summaries if we have more room
            remaining_tokens = max_tokens - result["total_tokens"]
            if remaining_tokens > 200:
                # Get consolidated summaries from DB
                summaries = await self._load_consolidated_summaries(user_id)
                for summary in summaries[:3]:  # Limit to 3 summaries
                    summary_tokens = len(str(summary)) // 4
                    if result["total_tokens"] + summary_tokens <= max_tokens:
                        result["topic_summaries"].append(summary)
                        result["total_tokens"] += summary_tokens

            self.logger.info(
                f"Hierarchical context for user {user_id}: "
                f"{len(result['recent_messages'])} recent, "
                f"{len(result['semantic_matches'])} semantic, "
                f"{len(result['topic_summaries'])} summaries, "
                f"~{result['total_tokens']} tokens"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving hierarchical context: {e}")
            return {
                "recent_messages": [],
                "semantic_matches": [],
                "topic_summaries": [],
                "total_tokens": 0,
            }

    async def summarize_conversation_segment(
        self,
        memories: List[ConversationMemory],
        max_summary_length: int = 500,
    ) -> str:
        """
        Create a summary of a conversation segment.
        Uses extractive summarization based on importance and topic coverage.
        """
        if not memories:
            return ""

        try:
            # Group by topic
            topic_contents: Dict[str, List[str]] = {}
            for memory in memories:
                topic = memory.topic or "general"
                if topic not in topic_contents:
                    topic_contents[topic] = []
                topic_contents[topic].append(memory.content[:200])  # First 200 chars

            # Build summary
            summary_parts = []
            for topic, contents in topic_contents.items():
                topic_summary = f"[{topic}]: {'; '.join(contents[:3])}"
                summary_parts.append(topic_summary)

            full_summary = " | ".join(summary_parts)
            
            # Truncate if too long
            if len(full_summary) > max_summary_length:
                full_summary = full_summary[:max_summary_length - 3] + "..."

            return full_summary

        except Exception as e:
            self.logger.error(f"Error summarizing conversation: {e}")
            return ""

    async def get_context_for_llm(
        self,
        user_id: int,
        query: str,
        max_context_chars: int = 32000,
    ) -> str:
        """
        Get optimized context string for LLM prompt injection.
        Combines hierarchical retrieval with formatting for LLM consumption.
        """
        try:
            hierarchical = await self.retrieve_hierarchical_context(
                user_id, query, max_tokens=max_context_chars // 4
            )

            context_parts = []

            # Add recent conversation
            if hierarchical["recent_messages"]:
                recent_text = "\n".join(
                    f"[{m.role}]: {m.content}" for m in hierarchical["recent_messages"]
                )
                context_parts.append(f"Recent conversation:\n{recent_text}")

            # Add semantically relevant memories
            if hierarchical["semantic_matches"]:
                relevant_text = "\n".join(
                    f"- {m.content[:300]}" for m in hierarchical["semantic_matches"][:5]
                )
                context_parts.append(f"Relevant context:\n{relevant_text}")

            # Add historical summaries
            if hierarchical["topic_summaries"]:
                summaries_text = "\n".join(
                    str(s.get("preview", s))[:200] 
                    for s in hierarchical["topic_summaries"]
                )
                context_parts.append(f"Historical context:\n{summaries_text}")

            full_context = "\n\n---\n\n".join(context_parts)
            
            # Ensure we don't exceed the limit
            if len(full_context) > max_context_chars:
                full_context = full_context[:max_context_chars - 3] + "..."

            return full_context

        except Exception as e:
            self.logger.error(f"Error getting context for LLM: {e}")
            return ""

    async def _load_consolidated_summaries(self, user_id: int) -> List[Dict[str, Any]]:
        """Load consolidated memory summaries from database"""
        try:
            if self.db is None:
                return []

            collection = self.db["consolidated_memories"]
            summaries = await asyncio.to_thread(
                lambda: list(collection.find({"user_id": user_id}).sort("consolidated_at", -1).limit(10))
            )
            return summaries

        except Exception as e:
            self.logger.error(f"Error loading consolidated summaries: {e}")
            return []

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
        """
        Calculate semantic similarity between query and content.
        Uses embedding-based similarity when LangChain is available,
        falls back to word overlap for basic similarity.
        """
        try:
            # Use embedding-based similarity if available
            if self.langchain_enabled and self.embeddings:
                try:
                    query_embedding = self.embeddings.embed_query(query)
                    content_embedding = self.embeddings.embed_query(content)
                    return self._cosine_similarity(query_embedding, content_embedding)
                except Exception as e:
                    self.logger.debug(f"Embedding similarity failed, using fallback: {e}")

            # Fallback: Simple word overlap similarity (Jaccard)
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
        profile_data = await self._load_profile_from_db(user_id)
        if not profile_data:
            return

        profile = UserProfile.from_dict(profile_data)
        current_score = profile.interests.get(topic, 0.0)
        new_score = min(current_score + 0.1, 1.0)
        profile.interests[topic] = new_score
        profile.last_updated = time.time()
        await self._save_profile_to_db(user_id, profile)

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

    async def _load_memories_from_db(self, user_id: int) -> List[ConversationMemory]:
        """Load user memories from database"""
        try:
            if self.db is None:
                return []

            collection = self.db["personalized_memories"]
            memories_data = await asyncio.to_thread(
                collection.find, {"user_id": user_id}
            )
            memories_data = list(memories_data)  # Convert cursor to list
            memories = [ConversationMemory(**data) for data in memories_data]
            return memories

        except Exception as e:
            self.logger.error(f"Error loading memories from DB: {e}")
            return []

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
