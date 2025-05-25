from src.extract.workflow_client import WorkflowRunner

runner = WorkflowRunner()

runner.run_batch(
    doc_type="receipt",
    local_folder="data/receipt",
    output_folder="data/receipts_output",
)

runner.run_batch(
    doc_type="payslip",
    local_folder="data/Income_statement",
    output_folder="data/payslips_output",
)
