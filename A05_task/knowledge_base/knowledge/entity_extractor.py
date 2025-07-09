"""Entity extraction module for the Knowledge Base System"""

import re
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from knowledge_base.knowledge import BaseKnowledgeOrganizer, KnowledgeConfig, EntityResult, RelationshipResult
from knowledge_base.models.schema import EntityType, RelationshipType


class EntityExtractor(BaseKnowledgeOrganizer):
    """Entity extractor and organizer"""
    
    def __init__(self):
        self.config = None
        self.entities = {}  # In-memory store of entities
        self.relationships = {}  # In-memory store of relationships
        self.entity_types = set()
        self.relationship_types = set()
        self.taxonomies = {}
    
    async def initialize(self, config: KnowledgeConfig) -> bool:
        """Initialize the entity extractor"""
        try:
            self.config = config
            
            # Set up entity types
            if config.entity_types:
                self.entity_types = set(config.entity_types)
            else:
                # Default entity types
                self.entity_types = {et.value for et in EntityType}
            
            # Set up relationship types
            if config.relationship_types:
                self.relationship_types = set(config.relationship_types)
            else:
                # Default relationship types
                self.relationship_types = {rt.value for rt in RelationshipType}
            
            # Load taxonomies if specified
            if config.taxonomy_file:
                await self._load_taxonomies(config.taxonomy_file)
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize entity extractor: {str(e)}")
            return False
    
    async def extract_entities(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        if not content or "text" not in content:
            return []
        
        text = content["text"]
        extracted_entities = []
        
        # Extract concepts (capitalized terms)
        concepts = self._extract_concepts(text)
        for concept in concepts:
            entity = {
                "id": f"ent_{uuid.uuid4().hex[:8]}",
                "name": concept,
                "entity_type": EntityType.CONCEPT.value,
                "confidence": 0.7,
                "source_id": content.get("source_id", ""),
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "extraction_method": "pattern_matching",
                }
            }
            extracted_entities.append(entity)
        
        # Extract technical terms
        terms = self._extract_technical_terms(text)
        for term in terms:
            entity = {
                "id": f"ent_{uuid.uuid4().hex[:8]}",
                "name": term,
                "entity_type": EntityType.TERM.value,
                "confidence": 0.8,
                "source_id": content.get("source_id", ""),
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "extraction_method": "pattern_matching",
                }
            }
            extracted_entities.append(entity)
        
        # Extract algorithms and methods
        algorithms = self._extract_algorithms(text)
        for algo in algorithms:
            entity = {
                "id": f"ent_{uuid.uuid4().hex[:8]}",
                "name": algo,
                "entity_type": EntityType.ALGORITHM.value,
                "confidence": 0.75,
                "source_id": content.get("source_id", ""),
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "extraction_method": "pattern_matching",
                }
            }
            extracted_entities.append(entity)
        
        # Deduplicate entities
        unique_entities = {}
        for entity in extracted_entities:
            name_lower = entity["name"].lower()
            if name_lower not in unique_entities or entity["confidence"] > unique_entities[name_lower]["confidence"]:
                unique_entities[name_lower] = entity
        
        return list(unique_entities.values())
    
    async def extract_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        if not entities or len(entities) < 2:
            return []
        
        relationships = []
        
        # Find similar entities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:  # Skip self-relationships and duplicates
                    continue
                
                # Check for similarity in names
                name1 = entity1["name"].lower()
                name2 = entity2["name"].lower()
                
                # Simple similarity check
                if (name1 in name2 or name2 in name1) and name1 != name2:
                    relationship = {
                        "id": f"rel_{uuid.uuid4().hex[:8]}",
                        "source_entity_id": entity1["id"],
                        "target_entity_id": entity2["id"],
                        "relationship_type": RelationshipType.SIMILAR_TO.value,
                        "confidence": 0.7,
                        "metadata": {
                            "extracted_at": datetime.now().isoformat(),
                            "extraction_method": "name_similarity",
                        }
                    }
                    relationships.append(relationship)
                
                # Check for taxonomy relationships
                if entity1["entity_type"] == entity2["entity_type"]:
                    # Check if they belong to the same taxonomy branch
                    taxonomy_relationship = self._check_taxonomy_relationship(entity1["name"], entity2["name"])
                    if taxonomy_relationship:
                        relationship = {
                            "id": f"rel_{uuid.uuid4().hex[:8]}",
                            "source_entity_id": entity1["id"],
                            "target_entity_id": entity2["id"],
                            "relationship_type": taxonomy_relationship,
                            "confidence": 0.8,
                            "metadata": {
                                "extracted_at": datetime.now().isoformat(),
                                "extraction_method": "taxonomy",
                            }
                        }
                        relationships.append(relationship)
        
        return relationships
    
    async def add_entity(self, entity: Dict[str, Any]) -> EntityResult:
        """Add an entity to the knowledge base"""
        try:
            # Generate ID if not present
            if "id" not in entity:
                entity["id"] = f"ent_{uuid.uuid4().hex[:8]}"
            
            # Add timestamps
            entity["created_at"] = datetime.now().isoformat()
            entity["updated_at"] = datetime.now().isoformat()
            
            # Store the entity
            self.entities[entity["id"]] = entity
            
            return EntityResult(
                success=True,
                entity_id=entity["id"],
                metadata={"entity_type": entity.get("entity_type", "")},
            )
            
        except Exception as e:
            return EntityResult(
                success=False,
                error=str(e),
            )
    
    async def add_relationship(self, relationship: Dict[str, Any]) -> RelationshipResult:
        """Add a relationship to the knowledge base"""
        try:
            # Check if the entities exist
            source_id = relationship.get("source_entity_id", "")
            target_id = relationship.get("target_entity_id", "")
            
            if source_id not in self.entities:
                return RelationshipResult(
                    success=False,
                    error=f"Source entity {source_id} not found",
                )
            
            if target_id not in self.entities:
                return RelationshipResult(
                    success=False,
                    error=f"Target entity {target_id} not found",
                )
            
            # Generate ID if not present
            if "id" not in relationship:
                relationship["id"] = f"rel_{uuid.uuid4().hex[:8]}"
            
            # Add timestamps
            relationship["created_at"] = datetime.now().isoformat()
            relationship["updated_at"] = datetime.now().isoformat()
            
            # Store the relationship
            self.relationships[relationship["id"]] = relationship
            
            return RelationshipResult(
                success=True,
                relationship_id=relationship["id"],
                metadata={
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "relationship_type": relationship.get("relationship_type", ""),
                },
            )
            
        except Exception as e:
            return RelationshipResult(
                success=False,
                error=str(e),
            )
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity from the knowledge base"""
        return self.entities.get(entity_id)
    
    async def find_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to the given entity"""
        if entity_id not in self.entities:
            return []
        
        related_entities = []
        visited = set()
        
        # Recursive function to find related entities
        def find_related(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            
            # Find relationships where this entity is the source or target
            for rel_id, rel in self.relationships.items():
                if rel["source_entity_id"] == current_id:
                    target_id = rel["target_entity_id"]
                    if target_id in self.entities and target_id not in visited:
                        entity = self.entities[target_id]
                        related_entities.append({
                            **entity,
                            "relationship": rel["relationship_type"],
                            "relationship_direction": "outgoing",
                        })
                        find_related(target_id, depth + 1)
                
                elif rel["target_entity_id"] == current_id:
                    source_id = rel["source_entity_id"]
                    if source_id in self.entities and source_id not in visited:
                        entity = self.entities[source_id]
                        related_entities.append({
                            **entity,
                            "relationship": rel["relationship_type"],
                            "relationship_direction": "incoming",
                        })
                        find_related(source_id, depth + 1)
        
        # Start the recursive search
        find_related(entity_id, 0)
        
        return related_entities
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract concepts from text (capitalized terms)"""
        # Find capitalized words that are likely concepts
        concept_pattern = r'\b[A-Z][a-z]{2,}\b'
        matches = re.findall(concept_pattern, text)
        
        # Filter out common capitalized words that are not concepts
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "I", "You", "He", "She", "It", "We", "They"}
        concepts = {match for match in matches if match not in common_words}
        
        return concepts
    
    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms from text"""
        # Find terms with specific patterns (e.g., camelCase, snake_case, hyphenated-terms)
        term_patterns = [
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # camelCase
            r'\b[a-z]+_[a-z]+(_[a-z]+)*\b',  # snake_case
            r'\b[a-z]+-[a-z]+(-[a-z]+)*\b',  # hyphenated-terms
        ]
        
        terms = set()
        for pattern in term_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        
        return terms
    
    def _extract_algorithms(self, text: str) -> Set[str]:
        """Extract algorithms and methods from text"""
        # Find terms that are likely algorithms or methods
        algorithm_patterns = [
            r'\b[A-Z][a-z]+ (Algorithm|Method)\b',
            r'\b[A-Z][a-z]+-[A-Z][a-z]+ (algorithm|method)\b',
        ]
        
        algorithms = set()
        for pattern in algorithm_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                algorithms.add(match[0])
        
        return algorithms
    
    async def _load_taxonomies(self, taxonomy_file: str) -> None:
        """Load taxonomies from a file"""
        # This would load taxonomies from a file
        # For now, we'll just use a simple placeholder
        self.taxonomies = {
            EntityType.CONCEPT.value: {
                "root_concepts": ["Machine Learning", "Data Science", "Artificial Intelligence"],
                "hierarchies": {
                    "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
                    "Supervised Learning": ["Classification", "Regression"],
                    "Unsupervised Learning": ["Clustering", "Dimensionality Reduction"],
                }
            }
        }
    
    def _check_taxonomy_relationship(self, name1: str, name2: str) -> Optional[str]:
        """Check if two entities have a taxonomy relationship"""
        # This is a simplified version that would be replaced with a more sophisticated implementation
        for entity_type, taxonomy in self.taxonomies.items():
            hierarchies = taxonomy.get("hierarchies", {})
            
            for parent, children in hierarchies.items():
                if name1 == parent and name2 in children:
                    return RelationshipType.HAS_PART.value
                elif name2 == parent and name1 in children:
                    return RelationshipType.PART_OF.value
        
        return None 