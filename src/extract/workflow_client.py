import os
import json
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from google.auth import default as google_auth_default
from google.cloud import storage
from google.cloud.workflows.executions_v1 import ExecutionsClient
from google.cloud.workflows.executions_v1.types import Execution

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
load_dotenv()


class WorkflowRunner:
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        self.region = os.getenv("WORKFLOW_REGION")
        self.workflow_name = os.getenv("WORKFLOW_NAME")
        self.processor_location = os.getenv("PROCESSOR_LOCATION", "eu")
        self.input_bucket = os.getenv("INPUT_BUCKET")
        self.output_bucket = os.getenv("OUTPUT_BUCKET")
        self.payslip_processor_id = os.getenv("PAYSLIP_PROCESSOR_ID")
        self.receipt_processor_id = os.getenv("RECEIPT_PROCESSOR_ID")

        if not all(
            [
                self.project_id,
                self.region,
                self.workflow_name,
                self.input_bucket,
                self.output_bucket,
            ]
        ):
            raise ValueError("Missing required environment variables in .env")

        self.credentials, _ = google_auth_default()
        self.client = ExecutionsClient(credentials=self.credentials)
        self.workflow_path = f"projects/{self.project_id}/locations/{self.region}/workflows/{self.workflow_name}"

    def trigger_workflow(self, input_payload: dict) -> str:
        logging.info(
            f"Sending input to workflow:\n{json.dumps(input_payload, indent=2)}"
        )
        logging.info(f"Triggering: {self.workflow_path}")
        execution = Execution(argument=json.dumps(input_payload))
        response = self.client.create_execution(
            parent=self.workflow_path, execution=execution
        )
        logging.info(f"Workflow started: {response.name}")
        return response.name

    def run_batch(self, doc_type: str, local_folder: str, output_folder: str):
        """
        Uploads a folder and triggers the workflow with docType = 'payslip' or 'receipt'.
        """
        if doc_type not in ["payslip", "receipt"]:
            raise ValueError("doc_type must be either 'payslip' or 'receipt'")

        run_id = uuid.uuid4().hex[:8]
        input_prefix = f"{doc_type}/input/{run_id}/"
        output_prefix = f"{doc_type}/output/{run_id}/"

        gcs = GCSBucketManager()
        gcs.upload_folder(local_folder, self.input_bucket, input_prefix)

        payload = {
            "inputBucket": self.input_bucket,
            "inputPrefix": input_prefix,
            "outputBucket": self.output_bucket,
            "outputPrefix": output_prefix,
            "docType": doc_type,
            "location": self.processor_location,
            "payslipProcessorId": self.payslip_processor_id,
            "receiptProcessorId": self.receipt_processor_id,
        }

        self.trigger_workflow(payload)
        gcs.download_folder(self.output_bucket, output_prefix, output_folder)


class GCSBucketManager:
    def __init__(self):
        self.client = storage.Client()

    def upload_folder(self, local_folder: str, bucket_name: str, prefix: str):
        bucket = self.client.bucket(bucket_name)
        local_path = Path(local_folder)

        for file in local_path.glob("*"):
            if file.is_file() and not file.name.startswith("."):
                blob = bucket.blob(f"{prefix}{file.name}")
                blob.upload_from_filename(str(file))
                logging.info(
                    f"Uploaded: {file} → gs://{bucket_name}/{prefix}{file.name}"
                )

    def download_folder(self, bucket_name: str, prefix: str, local_folder: str):
        local_path = Path(local_folder)
        local_path.mkdir(parents=True, exist_ok=True)

        blobs = self.client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            rel_path = blob.name.replace(prefix, "")
            if rel_path:
                destination = local_path / rel_path
                blob.download_to_filename(str(destination))
                logging.info(
                    f"Downloaded: gs://{bucket_name}/{blob.name} → {destination}"
                )
