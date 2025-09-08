"""
Structured output models for HyperRAG using Pydantic.
These models ensure reliable JSON schema adherence for LLM responses.
"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


# Keyword Extraction Models
class KeywordExtractionResponse(BaseModel):
    """Structured response for keyword extraction from queries"""
    high_level_keywords: List[str] = Field(
        description="Overarching concepts or themes in the query",
        min_items=1,
    )
    low_level_keywords: List[str] = Field(
        description="Specific entities, details, or concrete terms in the query",
        min_items=1,
    )


# Entity Extraction Models
class EntityType(str, Enum):
    """Standard entity types for extraction"""
    organization = "organization"
    person = "person"
    geo = "geo"
    event = "event"
    role = "role"
    concept = "concept"
    other = "other"


class Entity(BaseModel):
    """Represents an extracted entity"""
    entity_name: str = Field(
        description="Name of the entity in the same language as input text",
        min_items=1,
    )
    entity_type: EntityType = Field(
        description="Type classification of the entity",
        min_items=1,
    )
    entity_description: str = Field(
        description="Comprehensive description of the entity's attributes and activities",
        min_items=1,
    )
    additional_properties: Optional[str] = Field(
        default=None,
        description="Other attributes like time, space, emotion, motivation, etc."
    )


class LowOrderHyperedge(BaseModel):
    """Represents a pairwise relationship between two entities"""
    entity1: str = Field(description="Name of the first entity")
    entity2: str = Field(description="Name of the second entity")
    description: str = Field(
        description="Explanation of why these entities are related",
        min_items=1,
    )
    keywords: List[str] = Field(
        description="Keywords summarizing the relationship nature",
        min_items=1,
    )
    strength: float = Field(
        ge=0.0, le=1.0,
        description="Numerical score indicating relationship strength (0-1)",
    )


class HighOrderHyperedge(BaseModel):
    """Represents complex relationships among multiple entities"""
    entities: List[str] = Field(
        min_items=3,
        description="Collection of entity names in the high-order association",
    )
    description: str = Field(
        description="Detailed description covering all entities in the set",
        min_items=1,
    )
    generalization: str = Field(
        description="Concise summary of the entity set content",
        min_items=1,
    )
    keywords: List[str] = Field(
        description="Keywords summarizing the high-order association nature",   
        min_items=1,
    )
    strength: float = Field(
        ge=0.0, le=1.0,
        description="Numerical score indicating association strength (0-1)",
    )


class EntityExtractionResponse(BaseModel):
    """Complete structured response for entity extraction"""
    entities: List[Entity] = Field(
        description="All entities identified in the text",
        min_items=1,
    )
    low_order_hyperedges: List[LowOrderHyperedge] = Field(
        description="Pairwise relationships between entities",
        min_items=1,
    )
    high_level_keywords: List[str] = Field(
        description="Main ideas, concepts, or themes from the text",
        min_items=1,
    )
    high_order_hyperedges: List[HighOrderHyperedge] = Field(
        description="Complex multi-entity associations",
        min_items=1,
    )


# Simplified Entity Extraction for faster processing
class SimpleEntity(BaseModel):
    """Simplified entity for lite mode"""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    description: str = Field(description="Brief entity description")


class SimpleRelation(BaseModel):
    """Simplified relationship for lite mode"""
    entities: List[str] = Field(
        min_items=2,
        description="Related entity names",
    )
    description: str = Field(description="Relationship description")
    weight: float = Field(
        ge=0.0, le=1.0, default=0.75,
        description="Relationship weight",  
    )


class SimplifiedEntityExtractionResponse(BaseModel):
    """Simplified extraction response for lite mode"""
    entities: List[SimpleEntity] = Field(
        description="All entities identified in the text",
        min_items=1,
    )
    relations: List[SimpleRelation] = Field(
        description="Relationships between entities",
        min_items=1,
    )
    keywords: List[str] = Field(
        description="Main ideas, concepts, or themes from the text",
        min_items=1,
    )


# Helper function to convert structured output to the existing format
def convert_to_legacy_format(response: EntityExtractionResponse, 
                            tuple_delimiter: str = " | ",
                            record_delimiter: str = "\n") -> str:
    """
    Convert structured EntityExtractionResponse to the legacy string format
    used by the existing parsing logic.
    """
    lines = []
    
    # Add entities
    for entity in response.entities:
        parts = [
            '"Entity"',
            entity.entity_name,
            entity.entity_type.value,
            entity.entity_description,
            entity.additional_properties or ""
        ]
        lines.append(f"({tuple_delimiter.join(parts)})")
    
    # Add low-order hyperedges
    for edge in response.low_order_hyperedges:
        parts = [
            '"Low-order Hyperedge"',
            edge.entity1,
            edge.entity2,
            edge.description,
            ", ".join(edge.keywords),
            str(edge.strength)
        ]
        lines.append(f"({tuple_delimiter.join(parts)})")
    
    # Add high-level keywords
    if response.high_level_keywords:
        parts = ['"High-level keywords"', ", ".join(response.high_level_keywords)]
        lines.append(f"({tuple_delimiter.join(parts)})")
    
    # Add high-order hyperedges
    for hedge in response.high_order_hyperedges:
        parts = ['"High-order Hyperedge"'] + hedge.entities + [
            hedge.description,
            hedge.generalization,
            ", ".join(hedge.keywords),
            str(hedge.strength)
        ]
        lines.append(f"({tuple_delimiter.join(parts)})")
    
    return record_delimiter.join(lines)


def parse_legacy_to_structured(text: str,
                              tuple_delimiter: str = " | ",
                              record_delimiter: str = "\n") -> Optional[EntityExtractionResponse]:
    """
    Parse legacy format string back to structured EntityExtractionResponse.
    This is useful for backwards compatibility and testing.
    """
    try:
        entities = []
        low_order_hyperedges = []
        high_level_keywords = []
        high_order_hyperedges = []
        
        lines = text.strip().split(record_delimiter)
        
        for line in lines:
            if not line.strip():
                continue
                
            # Remove parentheses and parse
            if line.startswith("(") and line.endswith(")"):
                line = line[1:-1]
            
            parts = line.split(tuple_delimiter)
            if not parts:
                continue
                
            record_type = parts[0].strip('"')
            
            if record_type == "Entity" and len(parts) >= 5:
                entity = Entity(
                    entity_name=parts[1],
                    entity_type=EntityType(parts[2].lower()) if parts[2].lower() in [e.value for e in EntityType] else EntityType.other,
                    entity_description=parts[3],
                    additional_properties=parts[4] if parts[4] else None
                )
                entities.append(entity)
                
            elif record_type == "Low-order Hyperedge" and len(parts) >= 6:
                edge = LowOrderHyperedge(
                    entity1=parts[1],
                    entity2=parts[2],
                    description=parts[3],
                    keywords=parts[4].split(", "),
                    strength=float(parts[5]) if parts[5] else 0.75
                )
                low_order_hyperedges.append(edge)
                
            elif record_type == "High-level keywords" and len(parts) >= 2:
                high_level_keywords = parts[1].split(", ")
                
            elif record_type == "High-order Hyperedge" and len(parts) >= 5:
                # Find where the description starts (after all entity names)
                entity_count = len(parts) - 4  # Subtract type, desc, gen, keywords, strength
                hedge = HighOrderHyperedge(
                    entities=parts[1:entity_count],
                    description=parts[entity_count],
                    generalization=parts[entity_count + 1],
                    keywords=parts[entity_count + 2].split(", "),
                    strength=float(parts[entity_count + 3]) if parts[entity_count + 3] else 0.75
                )
                high_order_hyperedges.append(hedge)
        
        return EntityExtractionResponse(
            entities=entities,
            low_order_hyperedges=low_order_hyperedges,
            high_level_keywords=high_level_keywords,
            high_order_hyperedges=high_order_hyperedges
        )
    except Exception as e:
        print(f"Error parsing legacy format: {e}")
        return None