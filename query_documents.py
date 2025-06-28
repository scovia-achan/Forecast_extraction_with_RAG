from chroma_manager import *
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from prompt_template import *
from info_extractors import *
from dotenv import load_dotenv

def setup_query_engine():
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")  

    # Initialize models (same config as processing)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=api_key
    )
    
    llm = ChatOpenAI(  # Changed from GoogleGenerativeAI
        model="gpt-4o-mini",  # Changed from gemini-1.5-flash
        temperature=0.1,
        openai_api_key=api_key
    )
    
    # Load existing vector store
    chroma_manager = ChromaDBManager("./chroma_db")
    vectorstore = chroma_manager.load_existing_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=prompt_template
    )
    
    # Create query engine
    query_engine = DocumentQueryEngine(vectorstore, llm, prompt)
    return query_engine

def extract_content_from_response(response):
    """Helper function to extract text content from OpenAI AIMessage or other response types"""
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)

class FixedProjectionExtractor(ProjectionExtractor):
    """ProjectionExtractor with OpenAI AIMessage handling"""
    
    def _parse_response(self, response, source_doc) -> List[Dict]:
        """Parse LLM response and extract JSON - Fixed for OpenAI"""
        try:
            # Fix: Extract content from AIMessage if needed
            response_text = extract_content_from_response(response)
            
            # Find JSON array in response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                projections = json.loads(json_str)
                
                # Add source info
                for proj in projections:
                    proj['source_file'] = source_doc.metadata.get('filename', 'Unknown')
                
                return projections
            else:
                # Fallback: look for single JSON object
                obj_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if obj_match:
                    json_str = obj_match.group()
                    proj = json.loads(json_str)
                    proj['source_file'] = source_doc.metadata.get('filename', 'Unknown')
                    return [proj]
                
                return []
                
        except json.JSONDecodeError:
            response_text = extract_content_from_response(response)
            print(f"Could not parse JSON from: {response_text[:100]}...")
            return []
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []
    
    def _parse_batch_response(self, response, source_docs) -> List[Dict]:
        """Parse response from batch processing - Fixed for OpenAI"""
        try:
            # Fix: Extract content from AIMessage if needed
            response_text = extract_content_from_response(response)
                
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                projections = json.loads(json_str)
                
                # Add source info
                for proj in projections:
                    chunk_num = proj.get('chunk_number', '1')
                    try:
                        chunk_idx = int(chunk_num) - 1
                        if 0 <= chunk_idx < len(source_docs):
                            proj['source_file'] = source_docs[chunk_idx].metadata.get('filename', 'Unknown')
                        else:
                            proj['source_file'] = 'Unknown'
                    except:
                        proj['source_file'] = 'Unknown'
                
                return projections
            
            return []
                
        except json.JSONDecodeError:
            print(f"Could not parse JSON from batch response")
            return []
        except Exception as e:
            print(f"Error parsing batch response: {e}")
            return []

def main():
    query_engine = setup_query_engine()
    
    # Now you can query multiple times
    questions = [
        "In the text, what predictions have been made about Africa's digital transformation on Africa or connectivity and internet penetration in the region?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = query_engine.ask(
            topic="internet penetration or digital Transformation",
            place="Africa",
            year="2010 to 2025"
        )
        
        # Fix: Handle AIMessage response for regular queries too
        answer_text = extract_content_from_response(answer) if hasattr(answer, 'content') else answer
        print(f"A: {answer_text}")

    # NEW: Add projection extraction with fixed extractor
    print("\n" + "="*60)
    print("🎯 EXTRACTING STRUCTURED PROJECTIONS")
    print("="*60)
    
    # Create FIXED projection extractor using the same query engine
    extractor = FixedProjectionExtractor(query_engine)
    
    # Try different extraction methods
    print("🔍 Method 1: Comprehensive search...")
    projections1 = extractor.extract_projections()
    
    print(f"\n🔍 Method 2: Document-by-document analysis...")
    projections2 = extractor.extract_by_document()
    
    # Combine and deduplicate
    all_projections = projections1 + projections2
    final_projections = extractor._deduplicate(all_projections)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Method 1 found: {len(projections1)} projections")
    print(f"Method 2 found: {len(projections2)} projections") 
    print(f"Combined unique: {len(final_projections)} projections")
    
    # Save combined results
    extractor.save_to_csv(final_projections, "comprehensive_projections.csv")
    
    # Display a few examples
    if final_projections:
        print(f"\n📋 Sample projections found:")
        for i, proj in enumerate(final_projections[:5]):  # Show top 5
            print(f"\n{i+1}. {proj.get('projection', 'N/A')}")
            print(f"   📍 Entity: {proj.get('entity', 'N/A')}")
            print(f"   📅 Year Predicted: {proj.get('year_predicted', 'N/A')}")
            print(f"   🎯 Target Year: {proj.get('target_year', 'N/A')}")
            print(f"   📄 Source: {proj.get('source_file', 'N/A')}")
    else:
        print("No projections found. Check if documents contain prediction language.")


if __name__ == "__main__":
    main()