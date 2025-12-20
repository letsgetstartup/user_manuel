#!/bin/bash
# Trigger Immediate Indexing for ManualAI Data Store

PROJECT_ID="manualai-481406"
LOCATION="eu"
DATA_STORE_ID="manual02_1765869275504"
BUCKET_URI="gs://manualai02a/*"

echo "Triggering immediate import for $BUCKET_URI..."

curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://discoveryengine.googleapis.com/v1alpha/projects/$PROJECT_ID/locations/$LOCATION/collections/default_collection/dataStores/$DATA_STORE_ID/branches/0/documents:import" \
    -d '{
        "gcsSource": {
            "inputUris": ["'"$BUCKET_URI"'"],
            "dataSchema": "custom"
        },
        "reconciliationMode": "INCREMENTAL"
    }'

echo -e "\nImport request sent. Check Vertex AI Console for status."
