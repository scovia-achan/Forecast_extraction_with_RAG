# projection_extractor.py
import pandas as pd
import json
import re
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate

class ProjectionExtractor:
    """Extract ICT projections from documents using existing query engine"""
    
    def __init__(self, query_engine):
        """Initialize with existing DocumentQueryEngine"""
        self.query_engine = query_engine
        self.extraction_prompt = self._create_extraction_prompt()
    
    def _create_extraction_prompt(self) -> PromptTemplate:
        """Create specialized prompt for projection extraction"""
        template = """
You are analyzing text about ICT and digital transformation in Africa. Your task is to find ALL predictions, projections, forecasts, and targets mentioned in the text.

SEARCH FOR THESE PATTERNS:
- "will reach X% by YEAR"
- "projected to grow to X by YEAR" 
- "expected to achieve X% by YEAR"
- "targets of X% by YEAR"
- "forecasted to reach X by YEAR"
- "anticipated growth to X% by YEAR"
- "goals of X% connectivity by YEAR"
- Numbers with years (e.g., "50% by 2025", "$2 billion by 2030")

EXTRACT INFORMATION ABOUT:
✓ Internet/broadband penetration rates and targets
✓ Mobile phone/smartphone adoption predictions  
✓ Digital infrastructure development forecasts
✓ Technology adoption rate projections
✓ ICT investment predictions and funding targets
✓ Connectivity goals and timelines
✓ Digital economy growth forecasts
✓ Government digital transformation targets
✓ Industry growth projections
✓ User adoption milestones

For EACH projection found, extract exactly:
1. The specific prediction with numbers and timeline
2. Who made it (organization, report, study, expert, government)
3. When the prediction was made (year published/stated)
4. Target year (the year the prediction is for)

IMPORTANT: Look carefully through the ENTIRE text. Even if you find one projection, keep reading for more.

Format as JSON array (return empty array if none found):
[
  {{
    "projection": "specific prediction with numbers and timeline",
    "entity": "organization or source that made prediction",
    "year_predicted": "YYYY or 'unclear'",
    "target_year": "YYYY or 'not specified'"
  }}
]

Context about {topic} in {place} around {year}:
{context}

JSON array of ALL projections found:"""

        return PromptTemplate(
            input_variables=["topic", "place", "year", "context"],
            template=template
        )
    
    def extract_projections(self, topic: str = "ICT and digital transformation", 
                          place: str = "Africa", 
                          year: str = "2010 to 2025") -> List[Dict]:
        """Extract projections using comprehensive search strategy"""
        try:
            all_projections = []
            
            # Multiple search strategies for better coverage
            search_queries = [
                f"projections forecasts predictions {topic} {place}",
                f"targets goals objectives {topic} {place}",
                f"expected reach achieve {place} digital technology",
                f"by 2020 2025 2030 {place} internet mobile connectivity",
                f"penetration rate adoption {place} ICT technology",
                f"investment funding {place} digital infrastructure",
                f"growth trends {place} telecommunications broadband",
                f"ITU World Bank report {place} digital forecast"
            ]
            
            print(f"🔍 Running comprehensive search with {len(search_queries)} different queries...")
            
            all_docs = []
            for search_query in search_queries:
                print(f"   Searching: {search_query[:50]}...")
                
                # Get more documents per query
                docs = self.query_engine.retriever.get_relevant_documents(search_query)
                all_docs.extend(docs)
            
            # Remove duplicate documents based on content similarity
            unique_docs = self._deduplicate_documents(all_docs)
            print(f"📄 Found {len(unique_docs)} unique document chunks to analyze")
            
            # Process more chunks (increase from 8 to 20)
            chunks_to_process = min(20, len(unique_docs))
            
            for i, doc in enumerate(unique_docs[:chunks_to_process]):
                print(f"Processing chunk {i+1}/{chunks_to_process}...")
                
                # Use the extraction prompt with the document content
                formatted_prompt = self.extraction_prompt.format(
                    topic=topic,
                    place=place, 
                    year=year,
                    context=doc.page_content
                )
                
                response = self.query_engine.llm.invoke(formatted_prompt)
                projections = self._parse_response(response, doc)
                all_projections.extend(projections)
                
                # Show progress for found projections
                if projections:
                    print(f"   ✅ Found {len(projections)} projections in this chunk")
            
            # Remove duplicates
            unique_projections = self._deduplicate(all_projections)
            print(f"✅ Total unique projections found: {len(unique_projections)}")
            
            return unique_projections
            
        except Exception as e:
            print(f"Error extracting projections: {e}")
            return []
    
    def _parse_response(self, response: str, source_doc) -> List[Dict]:
        """Parse LLM response and extract JSON"""
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                projections = json.loads(json_str)
                
                # Add source info
                for proj in projections:
                    proj['source_file'] = source_doc.metadata.get('filename', 'Unknown')
                
                return projections
            else:
                # Fallback: look for single JSON object
                obj_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if obj_match:
                    json_str = obj_match.group()
                    proj = json.loads(json_str)
                    proj['source_file'] = source_doc.metadata.get('filename', 'Unknown')
                    return [proj]
                
                return []
                
        except json.JSONDecodeError:
            print(f"Could not parse JSON from: {response[:100]}...")
            return []
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []
    
    def _deduplicate_documents(self, documents) -> List:
        """Remove duplicate documents based on content similarity"""
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Use first 200 characters as fingerprint
            content_hash = hash(doc.page_content[:200].strip().lower())
            
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
        
        return unique_docs
    def _deduplicate(self, projections: List[Dict]) -> List[Dict]:
        """Remove duplicate projections with better similarity detection"""
        seen = set()
        unique = []
        
        for proj in projections:
            # Create more detailed fingerprint
            projection_text = proj.get('projection', '').lower().strip()
            entity = proj.get('entity', '').lower().strip()
            target_year = proj.get('target_year', '').strip()
            
            # Extract numbers from projection for better matching
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', projection_text)
            numbers_str = ''.join(numbers)
            
            fingerprint = (
                projection_text[:100],  # First 100 chars
                entity,
                target_year,
                numbers_str  # Include extracted numbers
            )
            
            if fingerprint not in seen and projection_text:
                seen.add(fingerprint)
                unique.append(proj)
            elif projection_text:
                print(f"   Duplicate removed: {projection_text[:50]}...")
        
        return unique
    
    def save_to_csv(self, projections: List[Dict], filename: str = "africa_ict_projections.csv"):
        """Save projections to CSV file"""
        if not projections:
            print("No projections to save")
            return
        
        df = pd.DataFrame(projections)
        
        # Reorder columns for readability
        column_order = ['projection', 'entity', 'year_predicted', 'target_year', 'source_file']
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"📊 Saved {len(projections)} projections to {filename}")
        
        # Print summary
        print(f"\nSummary:")
        if not df.empty:
            print(f"- Total projections: {len(df)}")
            if 'entity' in df.columns:
                print(f"- Top entities: {df['entity'].value_counts().head(3).to_dict()}")
            if 'target_year' in df.columns:
                print(f"- Target years: {df['target_year'].value_counts().head().to_dict()}")
    
    def extract_by_document(self) -> List[Dict]:
        """Alternative method: Extract from each document individually"""
        try:
            # Get all unique documents from your vector store
            all_docs_query = "Africa ICT digital transformation internet mobile"
            docs = self.query_engine.retriever.get_relevant_documents(all_docs_query)
            
            # Group by source document
            docs_by_file = {}
            for doc in docs:
                filename = doc.metadata.get('filename', 'unknown')
                if filename not in docs_by_file:
                    docs_by_file[filename] = []
                docs_by_file[filename].append(doc)
            
            print(f"📚 Processing {len(docs_by_file)} different documents...")
            
            all_projections = []
            
            for filename, file_docs in docs_by_file.items():
                print(f"\n📄 Processing: {filename}")
                
                # Combine chunks from same document for better context
                combined_content = "\n\n".join([doc.page_content for doc in file_docs[:5]])  # Top 5 chunks per doc
                
                # Create a combined document object
                combined_doc = type('obj', (object,), {
                    'page_content': combined_content,
                    'metadata': {'filename': filename}
                })
                
                formatted_prompt = self.extraction_prompt.format(
                    topic="ICT and digital transformation",
                    place="Africa",
                    year="2010 to 2030",
                    context=combined_content
                )
                
                response = self.query_engine.llm.invoke(formatted_prompt)
                projections = self._parse_response(response, combined_doc)
                
                if projections:
                    print(f"   ✅ Found {len(projections)} projections")
                    all_projections.extend(projections)
                else:
                    print(f"   ❌ No projections found")
            
            unique_projections = self._deduplicate(all_projections)
            print(f"\n🎯 Total unique projections: {len(unique_projections)}")
            
            return unique_projections
            
        except Exception as e:
            print(f"Error in document-by-document extraction: {e}")
            return []
    def quick_extract(self, filename: str = "projections.csv", method: str = "comprehensive") -> List[Dict]:
        """One-liner to extract and save projections with different methods"""
        if method == "comprehensive":
            projections = self.extract_projections()
        elif method == "by_document":
            projections = self.extract_by_document()
        else:
            # Combine both methods
            proj1 = self.extract_projections()
            proj2 = self.extract_by_document()
            combined = proj1 + proj2
            projections = self._deduplicate(combined)
            print(f"🔄 Combined method found {len(projections)} unique projections")
        
        self.save_to_csv(projections, filename)
        return projections