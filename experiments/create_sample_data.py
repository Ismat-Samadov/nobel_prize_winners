#!/usr/bin/env python3
"""
Create sample CoNLL format data for testing the active learning NER system.

This script generates synthetic NER data in CoNLL format for demonstration purposes.
"""

import random
import os
from typing import List, Tuple


class SampleDataGenerator:
    """Generate sample NER data in CoNLL format."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        
        # Sample entities and contexts
        self.persons = [
            "John Smith", "Mary Johnson", "David Wilson", "Sarah Brown", "Michael Davis",
            "Jennifer Garcia", "Robert Miller", "Lisa Anderson", "William Taylor", "Emily Moore"
        ]
        
        self.organizations = [
            "Microsoft", "Google", "Apple", "Facebook", "Amazon", "Tesla", "Netflix",
            "Stanford University", "MIT", "Harvard University", "OpenAI", "DeepMind"
        ]
        
        self.locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "United States", "California", "Texas", "New York", "Florida", "Illinois"
        ]
        
        self.misc_entities = [
            "Python", "JavaScript", "React", "TensorFlow", "PyTorch", "COVID-19",
            "Nobel Prize", "Olympics", "World Cup", "Grammy Awards"
        ]
        
        # Sample sentence templates
        self.sentence_templates = [
            "{person} works at {organization} in {location}.",
            "{person} is the CEO of {organization}.",
            "{organization} was founded in {location}.",
            "The conference will be held in {location} next month.",
            "{person} won the {misc} award last year.",
            "{organization} released a new version of {misc}.",
            "Researchers at {organization} developed {misc}.",
            "{person} moved from {location} to {location}.",
            "The {misc} project is supported by {organization}.",
            "{person} studied at {organization} in {location}."
        ]
    
    def _get_entity_tokens(self, entity: str, entity_type: str) -> List[Tuple[str, str]]:
        """Convert entity to tokens with BIO tags."""
        tokens = entity.split()
        if len(tokens) == 1:
            return [(tokens[0], f"B-{entity_type}")]
        else:
            result = [(tokens[0], f"B-{entity_type}")]
            for token in tokens[1:]:
                result.append((token, f"I-{entity_type}"))
            return result
    
    def generate_sentence(self) -> List[Tuple[str, str]]:
        """Generate a single sentence with NER tags."""
        template = random.choice(self.sentence_templates)
        
        # Track which placeholders are in the template
        entities_used = {}
        
        if "{person}" in template:
            entities_used["person"] = random.choice(self.persons)
        if "{organization}" in template:
            entities_used["organization"] = random.choice(self.organizations)
        if "{location}" in template:
            entities_used["location"] = random.choice(self.locations)
        if "{misc}" in template:
            entities_used["misc"] = random.choice(self.misc_entities)
        
        # Handle multiple locations in same sentence
        location_count = template.count("{location}")
        if location_count > 1:
            locations = random.sample(self.locations, min(location_count, len(self.locations)))
            template_parts = template.split("{location}")
            sentence = template_parts[0]
            for i, loc in enumerate(locations):
                sentence += loc
                if i < len(template_parts) - 1:
                    sentence += template_parts[i + 1]
            entities_used["locations"] = locations
        else:
            sentence = template.format(**entities_used)
        
        # Tokenize and create labels
        tokens_with_labels = []
        words = sentence.replace(".", " .").replace(",", " ,").split()
        
        for word in words:
            found_entity = False
            
            # Check if word is part of any entity
            for entity_type, entity_value in entities_used.items():
                if entity_type == "locations":  # Handle multiple locations
                    for location in entity_value:
                        if word in location.split():
                            entity_tokens = self._get_entity_tokens(location, "LOC")
                            for token, label in entity_tokens:
                                if token == word:
                                    tokens_with_labels.append((word, label))
                                    found_entity = True
                                    break
                            if found_entity:
                                break
                else:
                    if isinstance(entity_value, str) and word in entity_value.split():
                        # Determine entity type
                        if entity_type == "person":
                            ner_type = "PER"
                        elif entity_type == "organization":
                            ner_type = "ORG"
                        elif entity_type == "location":
                            ner_type = "LOC"
                        else:
                            ner_type = "MISC"
                        
                        entity_tokens = self._get_entity_tokens(entity_value, ner_type)
                        for token, label in entity_tokens:
                            if token == word:
                                tokens_with_labels.append((word, label))
                                found_entity = True
                                break
                
                if found_entity:
                    break
            
            if not found_entity:
                tokens_with_labels.append((word, "O"))
        
        return tokens_with_labels
    
    def generate_dataset(self, num_sentences: int) -> List[List[Tuple[str, str]]]:
        """Generate a dataset with multiple sentences."""
        dataset = []
        for _ in range(num_sentences):
            sentence = self.generate_sentence()
            dataset.append(sentence)
        return dataset
    
    def save_conll_format(self, dataset: List[List[Tuple[str, str]]], filename: str):
        """Save dataset in CoNLL format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for sentence in dataset:
                for token, label in sentence:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Empty line between sentences


def main():
    """Generate sample datasets for training and testing."""
    
    # Create data directory
    os.makedirs("../data", exist_ok=True)
    
    # Initialize generator
    generator = SampleDataGenerator(seed=42)
    
    # Generate training data (larger dataset)
    print("Generating training data...")
    train_dataset = generator.generate_dataset(1000)
    generator.save_conll_format(train_dataset, "../data/train.conll")
    print(f"Generated {len(train_dataset)} training sentences")
    
    # Generate test data (smaller dataset)
    print("Generating test data...")
    test_dataset = generator.generate_dataset(200)
    generator.save_conll_format(test_dataset, "../data/test.conll")
    print(f"Generated {len(test_dataset)} test sentences")
    
    # Print sample data
    print("\nSample training data:")
    for i, sentence in enumerate(train_dataset[:3]):
        print(f"Sentence {i+1}:")
        for token, label in sentence:
            print(f"  {token}\t{label}")
        print()
    
    print("Sample data generation completed!")
    print("Files created:")
    print("  - ../data/train.conll")
    print("  - ../data/test.conll")


if __name__ == "__main__":
    main()