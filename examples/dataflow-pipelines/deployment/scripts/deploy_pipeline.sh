#!/bin/bash

set -euo pipefail

# Google Dataflow Pipeline Deployment Script
# 
# This script deploys either batch or streaming Dataflow pipelines with proper
# configuration and error handling.
#
# Usage:
#   ./deploy_pipeline.sh <pipeline_type> <pipeline_name> <environment> [additional_args...]
#
# Examples:
#   ./deploy_pipeline.sh batch user_events dev
#   ./deploy_pipeline.sh streaming real_time_events prod --max_num_workers=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
DEFAULT_REGION="us-central1"
DEFAULT_ZONE="us-central1-a"
DEFAULT_WORKER_MACHINE_TYPE="n1-standard-2"
DEFAULT_MAX_WORKERS=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "No active gcloud authentication found. Please run 'gcloud auth login'"
        exit 1
    fi
    
    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        log_warn "uv is not installed. Falling back to pip for dependency management"
    fi
    
    # Check if Python 3.9+ is available
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        log_error "Python 3.9 or higher is required"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Function to validate arguments
validate_arguments() {
    if [[ $# -lt 3 ]]; then
        log_error "Usage: $0 <pipeline_type> <pipeline_name> <environment> [additional_args...]"
        echo
        echo "Pipeline types: batch, streaming"
        echo "Pipeline names: user_events, transaction_data, audit_logs, real_time_events, clickstream_analytics, iot_sensors, fraud_detection"
        echo "Environments: dev, staging, prod"
        exit 1
    fi
    
    PIPELINE_TYPE="$1"
    PIPELINE_NAME="$2"
    ENVIRONMENT="$3"
    shift 3
    ADDITIONAL_ARGS=("$@")
    
    # Validate pipeline type
    if [[ ! "$PIPELINE_TYPE" =~ ^(batch|streaming)$ ]]; then
        log_error "Invalid pipeline type: $PIPELINE_TYPE. Must be 'batch' or 'streaming'"
        exit 1
    fi
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be 'dev', 'staging', or 'prod'"
        exit 1
    fi
    
    # Check if pipeline exists
    PIPELINE_PATH="$PROJECT_ROOT/pipelines/$PIPELINE_TYPE/$PIPELINE_NAME"
    if [[ ! -d "$PIPELINE_PATH" ]]; then
        log_error "Pipeline not found: $PIPELINE_PATH"
        exit 1
    fi
    
    log_info "Arguments validated successfully"
}

# Function to load configuration
load_configuration() {
    log_info "Loading configuration for $ENVIRONMENT environment..."
    
    # Load environment-specific configuration
    CONFIG_FILE="$PROJECT_ROOT/deployment/configs/$ENVIRONMENT.json"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Extract configuration values using Python
    PROJECT_ID=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('project_id', ''))
")
    
    REGION=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('region', '$DEFAULT_REGION'))
")
    
    TEMP_LOCATION=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('temp_location', 'gs://${PROJECT_ID}-dataflow-temp/'))
" | envsubst)
    
    STAGING_LOCATION=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('staging_location', 'gs://${PROJECT_ID}-dataflow-staging/'))
" | envsubst)
    
    # Validate required configuration
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "project_id not found in configuration file"
        exit 1
    fi
    
    log_info "Configuration loaded: PROJECT_ID=$PROJECT_ID, REGION=$REGION"
}

# Function to create GCS buckets if they don't exist
create_gcs_buckets() {
    log_info "Ensuring required GCS buckets exist..."
    
    # Extract bucket names from paths
    TEMP_BUCKET=$(echo "$TEMP_LOCATION" | sed 's|gs://\([^/]*\)/.*|\1|')
    STAGING_BUCKET=$(echo "$STAGING_LOCATION" | sed 's|gs://\([^/]*\)/.*|\1|')
    
    # Create buckets if they don't exist
    for bucket in "$TEMP_BUCKET" "$STAGING_BUCKET"; do
        if ! gsutil ls "gs://$bucket" &> /dev/null; then
            log_info "Creating GCS bucket: $bucket"
            gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$bucket"
        else
            log_info "GCS bucket already exists: $bucket"
        fi
    done
}

# Function to install dependencies
install_dependencies() {
    log_info "Installing pipeline dependencies..."
    
    cd "$PROJECT_ROOT"
    
    if command -v uv &> /dev/null; then
        log_info "Using uv for dependency management"
        uv sync --all-extras
    else
        log_info "Using pip for dependency management"
        pip install -e .
        pip install -e common/
        pip install -e "pipelines/$PIPELINE_TYPE/$PIPELINE_NAME/"
    fi
}

# Function to run tests
run_tests() {
    if [[ "$ENVIRONMENT" != "prod" ]]; then
        log_info "Running pipeline tests..."
        
        cd "$PROJECT_ROOT"
        
        if command -v uv &> /dev/null; then
            uv run pytest "pipelines/$PIPELINE_TYPE/$PIPELINE_NAME/tests/" -v
        else
            pytest "pipelines/$PIPELINE_TYPE/$PIPELINE_NAME/tests/" -v
        fi
        
        log_info "Tests passed successfully"
    else
        log_warn "Skipping tests for production deployment"
    fi
}

# Function to deploy batch pipeline
deploy_batch_pipeline() {
    log_info "Deploying batch pipeline: $PIPELINE_NAME"
    
    PIPELINE_SCRIPT="$PROJECT_ROOT/pipelines/batch/$PIPELINE_NAME/src/$PIPELINE_NAME/pipeline.py"
    
    if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
        log_error "Pipeline script not found: $PIPELINE_SCRIPT"
        exit 1
    fi
    
    # Generate job name with timestamp
    JOB_NAME="${PIPELINE_NAME}-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    
    # Build deployment command
    DEPLOY_CMD=(
        python3 "$PIPELINE_SCRIPT"
        --runner=DataflowRunner
        --project="$PROJECT_ID"
        --region="$REGION"
        --temp_location="$TEMP_LOCATION"
        --staging_location="$STAGING_LOCATION"
        --setup_file="$PROJECT_ROOT/setup.py"
        --job_name="$JOB_NAME"
        --save_main_session
        --config_file="$PROJECT_ROOT/deployment/configs/$ENVIRONMENT.json"
    )
    
    # Add environment-specific settings
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        DEPLOY_CMD+=(
            --max_num_workers=20
            --autoscaling_algorithm=THROUGHPUT_BASED
            --enable_streaming_engine
        )
    else
        DEPLOY_CMD+=(
            --max_num_workers=5
        )
    fi
    
    # Add any additional arguments
    DEPLOY_CMD+=("${ADDITIONAL_ARGS[@]}")
    
    log_info "Executing deployment command..."
    log_info "Job name: $JOB_NAME"
    
    "${DEPLOY_CMD[@]}"
    
    log_info "Batch pipeline deployment initiated successfully"
    log_info "Monitor job at: https://console.cloud.google.com/dataflow/jobs/$REGION/$JOB_NAME?project=$PROJECT_ID"
}

# Function to deploy streaming pipeline
deploy_streaming_pipeline() {
    log_info "Deploying streaming pipeline: $PIPELINE_NAME"
    
    PIPELINE_SCRIPT="$PROJECT_ROOT/pipelines/streaming/$PIPELINE_NAME/src/$PIPELINE_NAME/pipeline.py"
    
    if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
        log_error "Pipeline script not found: $PIPELINE_SCRIPT"
        exit 1
    fi
    
    # Generate job name with timestamp
    JOB_NAME="${PIPELINE_NAME}-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    
    # Build deployment command
    DEPLOY_CMD=(
        python3 "$PIPELINE_SCRIPT"
        --runner=DataflowRunner
        --project="$PROJECT_ID"
        --region="$REGION"
        --temp_location="$TEMP_LOCATION"
        --staging_location="$STAGING_LOCATION"
        --setup_file="$PROJECT_ROOT/setup.py"
        --job_name="$JOB_NAME"
        --streaming
        --enable_streaming_engine
        --save_main_session
        --config_file="$PROJECT_ROOT/deployment/configs/$ENVIRONMENT.json"
    )
    
    # Add environment-specific settings
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        DEPLOY_CMD+=(
            --max_num_workers=50
            --autoscaling_algorithm=THROUGHPUT_BASED
            --worker_machine_type=n1-standard-4
        )
    else
        DEPLOY_CMD+=(
            --max_num_workers=10
            --worker_machine_type=n1-standard-2
        )
    fi
    
    # Add any additional arguments
    DEPLOY_CMD+=("${ADDITIONAL_ARGS[@]}")
    
    log_info "Executing deployment command..."
    log_info "Job name: $JOB_NAME"
    
    "${DEPLOY_CMD[@]}"
    
    log_info "Streaming pipeline deployment initiated successfully"
    log_info "Monitor job at: https://console.cloud.google.com/dataflow/jobs/$REGION/$JOB_NAME?project=$PROJECT_ID"
}

# Function to create Dataflow Flex Template
create_flex_template() {
    if [[ "${CREATE_TEMPLATE:-}" == "true" ]]; then
        log_info "Creating Dataflow Flex Template..."
        
        TEMPLATE_IMAGE="gcr.io/$PROJECT_ID/$PIPELINE_NAME-template:latest"
        TEMPLATE_PATH="gs://$PROJECT_ID-dataflow-templates/$PIPELINE_TYPE/$PIPELINE_NAME/template.json"
        
        # Build Docker image for template
        cat > "$PROJECT_ROOT/Dockerfile.template" <<EOF
FROM gcr.io/dataflow-templates-base/python3-template-launcher-base

ENV FLEX_TEMPLATE_PYTHON_PY_FILE="pipelines/$PIPELINE_TYPE/$PIPELINE_NAME/src/$PIPELINE_NAME/pipeline.py"

COPY . /template
WORKDIR /template

RUN pip install -e .
RUN pip install -e common/
RUN pip install -e pipelines/$PIPELINE_TYPE/$PIPELINE_NAME/

ENV FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE="requirements.txt"
EOF
        
        # Build and push image
        gcloud builds submit --tag "$TEMPLATE_IMAGE" .
        
        # Create template spec
        gcloud dataflow flex-template build "$TEMPLATE_PATH" \
            --image "$TEMPLATE_IMAGE" \
            --sdk-language "PYTHON"
        
        log_info "Flex Template created: $TEMPLATE_PATH"
        
        # Clean up
        rm -f "$PROJECT_ROOT/Dockerfile.template"
    fi
}

# Function to validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Wait a moment for the job to start
    sleep 30
    
    # Check if job is running (for the last job created)
    RECENT_JOB=$(gcloud dataflow jobs list \
        --filter="name~$PIPELINE_NAME AND region:$REGION" \
        --sort-by="~createTime" \
        --limit=1 \
        --format="value(JOB_ID)" 2>/dev/null)
    
    if [[ -n "$RECENT_JOB" ]]; then
        JOB_STATE=$(gcloud dataflow jobs describe "$RECENT_JOB" \
            --region="$REGION" \
            --format="value(currentState)" 2>/dev/null)
        
        log_info "Job ID: $RECENT_JOB"
        log_info "Job State: $JOB_STATE"
        
        if [[ "$JOB_STATE" == "JOB_STATE_RUNNING" ]]; then
            log_info "Pipeline is running successfully"
        elif [[ "$JOB_STATE" == "JOB_STATE_PENDING" ]]; then
            log_info "Pipeline is starting up"
        else
            log_warn "Pipeline state: $JOB_STATE"
        fi
    else
        log_warn "Could not verify job status"
    fi
}

# Main function
main() {
    log_info "Starting Dataflow pipeline deployment..."
    log_info "Pipeline: $PIPELINE_TYPE/$PIPELINE_NAME"
    log_info "Environment: $ENVIRONMENT"
    
    check_prerequisites
    validate_arguments "$@"
    load_configuration
    create_gcs_buckets
    install_dependencies
    run_tests
    create_flex_template
    
    # Deploy based on pipeline type
    if [[ "$PIPELINE_TYPE" == "batch" ]]; then
        deploy_batch_pipeline
    else
        deploy_streaming_pipeline
    fi
    
    validate_deployment
    
    log_info "Deployment completed successfully!"
    log_info "View your pipeline: https://console.cloud.google.com/dataflow?project=$PROJECT_ID"
}

# Check if script is being sourced or executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi