from google.cloud import discoveryengine_v1beta as discoveryengine
from google.api_core.client_options import ClientOptions
import time

def setup_automated_data_store(project_id, location, bucket_name):
    # 1. Initialize the Client
    # Client options can be useful if we need to specify api_endpoint
    # Explicitly set quota_project_id to satisfy API requirements
    client_options = ClientOptions(quota_project_id=project_id)
    client = discoveryengine.DataStoreServiceClient(client_options=client_options)

    # 2. Define the Data Store configuration
    data_store = discoveryengine.DataStore(
        display_name="Manuals_Data_Store",
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
        solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_CHAT],
    )

    # 3. Create the Data Store
    # The parent resource name: projects/{project}/locations/{location}/collections/{collection}
    parent = f"projects/{project_id}/locations/{location}/collections/default_collection"
    data_store_id = "manualai02a-automated-store"
    
    print(f"Creating Data Store '{data_store_id}' in '{parent}'...")
    
    try:
        operation = client.create_data_store(
            parent=parent,
            data_store=data_store,
            data_store_id=data_store_id
        )
        print("Waiting for Data Store creation to complete...")
        response = operation.result()
        print(f"Data Store created: {response.name}")
    except Exception as e:
        if "409" in str(e) and "already exists" in str(e):
             print(f"Data store {data_store_id} already exists, proceeding to import.")
        else:
            raise e

    # 4. Import the files from your bucket
    # We need to wait a bit or just ensure the store is ready. 
    # The operation.result() blocks until done, so strictly speaking we are ready.

    import_client = discoveryengine.DocumentServiceClient()
    
    # The parent for import is the branch created inside the data store.
    # default branch is usually '0' (for default_branch)
    import_parent = f"projects/{project_id}/locations/{location}/collections/default_collection/dataStores/{data_store_id}/branches/0"
    
    import_request = discoveryengine.ImportDocumentsRequest(
        parent=import_parent,
        gcs_source=discoveryengine.GcsSource(
            input_uris=[f"gs://{bucket_name}/*.pdf"]
        ),
        # This setting ensures the system looks for unstructured PDF content
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    )

    print(f"Starting import from gs://{bucket_name}/*.pdf...")
    import_op = import_client.import_documents(request=import_request)
    print(f"Import operation started: {import_op.operation.name}")
    print("Waiting for import to complete...")
    
    # We can block and wait for the result
    import_response = import_op.result()
    print("Import completed successfully.")
    print(import_response)

# Execution variables
PROJECT_ID = "manualai" 
LOCATION = "us-central1"
BUCKET_NAME = "manualai02a"

if __name__ == "__main__":
    setup_automated_data_store(PROJECT_ID, LOCATION, BUCKET_NAME)
