from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google import genai
import os

PROJECT_ID = os.environ.get("PROJECT_ID", "manualai-481406")
LOCATION = "us-central1"

class GraphSearcher:
    def __init__(self):
        self.db = firestore.Client(project=PROJECT_ID)
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        # Using the same model as in ingestion
        self.embedding_model = "text-embedding-004"

    def embed_query(self, text):
        """Turns user question into a vector."""
        # Using the new genai SDK style
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text,
        )
        # result.embeddings is a list of ContentEmbedding objects
        # We need the values of the first embedding for the query
        return result.embeddings[0].values

    def find_solution_node(self, machine_id, user_query):
        """
        Finds the most relevant starting node in the graph using Vector Search.
        """
        query_vector = self.embed_query(user_query)
        
        # Reference to the nodes collection for the specific machine
        # Path: machines/{machine_id}/graph_nodes
        nodes_ref = self.db.collection("machines").document(machine_id).collection("graph_nodes")
        
        # Perform Vector Search (Nearest Neighbor)
        # Note: Requires a Vector Index created in Firestore
        vector_query = nodes_ref.find_nearest(
            vector_field="embedding_field",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=1
        )
        
        results = vector_query.get()
        
        if not results:
            return None

        # Get the best match
        best_node = results[0]
        node_data = best_node.to_dict()
        
        # Fetch connected edges to know "what's next"
        # Path: machines/{machine_id}/graph_edges
        edges_ref = self.db.collection("machines").document(machine_id).collection("graph_edges")
        
        # Query for edges where this node is the 'from' node
        next_steps = edges_ref.where("from", "==", node_data['id']).stream()
        
        node_data['next_possible_steps'] = [edge.to_dict() for edge in next_steps]
        
        return node_data

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Ensure you set PROJECT_ID if not using default
    # os.environ["PROJECT_ID"] = "manualai-481406"
    MODEL_NAME = "gemini-2.5-pro"
    
    searcher = GraphSearcher()
    
    # 1. User asks a question
    # Example query
    query = "The steam wand isn't making any foam"
    
    # Example machine ID (must match what was processed in Cloud Function)
    # If you uploaded "bes875-instruction-manual.pdf", the ID is "bes875-instruction-manual"
    machine = "bes875-instruction-manual" 
    
    print(f"🔍 Searching graph for: '{query}'...")
    try:
        result = searcher.find_solution_node(machine, query)
        
        if result:
            print("\n--- 🎯 FOUND NODE ---")
            print(f"Step: {result.get('label')}")
            print(f"Action: {result.get('text')}")
            print(f"AR Anchor: {result.get('ar_anchor', 'None')}")
            
            print("\n--- ➡️ NEXT LOGICAL STEPS ---")
            steps = result.get('next_possible_steps', [])
            if steps:
                for step in steps:
                    print(f"- If {step.get('condition')}, go to -> {step.get('to')}")
            else:
                 print("No further steps found (End of path or data issue).")
        else:
            print("No solution found in graph or machine ID might be incorrect.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Tip: Check if Firestore Vector Index is created and machine ID exists.")
