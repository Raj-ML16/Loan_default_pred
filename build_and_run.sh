#!/bin/bash
# build_and_run.sh - Build and run your ML API with Podman

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="loan-prediction-api"
IMAGE_TAG="latest"
CONTAINER_NAME="loan-api-container"
PORT=8000

echo -e "${BLUE}🚀 MLOps Loan Prediction API - Build & Run Script${NC}"
echo -e "${BLUE}=================================================${NC}"

# Function to check if Podman is installed
check_podman() {
    if ! command -v podman &> /dev/null; then
        echo -e "${RED}❌ Podman is not installed!${NC}"
        echo -e "${YELLOW}📦 Install Podman:${NC}"
        echo -e "${YELLOW}   - Ubuntu/Debian: sudo apt install podman${NC}"
        echo -e "${YELLOW}   - CentOS/RHEL: sudo dnf install podman${NC}"
        echo -e "${YELLOW}   - macOS: brew install podman${NC}"
        echo -e "${YELLOW}   - Windows: https://podman.io/getting-started/installation${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Podman is installed${NC}"
}

# Function to build image
build_image() {
    echo -e "${BLUE}🔨 Building container image...${NC}"
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}❌ Dockerfile not found!${NC}"
        exit 1
    fi
    
    # Build the image
    podman build -t ${IMAGE_NAME}:${IMAGE_TAG} . --no-cache
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    else
        echo -e "${RED}❌ Build failed!${NC}"
        exit 1
    fi
}

# Function to stop and remove existing container
cleanup_container() {
    echo -e "${YELLOW}🧹 Cleaning up existing container...${NC}"
    
    # Stop container if running
    podman stop ${CONTAINER_NAME} 2>/dev/null
    
    # Remove container if exists
    podman rm ${CONTAINER_NAME} 2>/dev/null
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Function to run container
run_container() {
    echo -e "${BLUE}🚀 Starting container...${NC}"
    
    # Run the container
    podman run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8000 \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/model_predictions.db:/app/model_predictions.db \
        --health-cmd="python -c 'import requests; requests.get(\"http://localhost:8000/health\")'" \
        --health-interval=30s \
        --health-retries=3 \
        --health-start-period=10s \
        ${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Container started successfully!${NC}"
        echo -e "${BLUE}📊 API Endpoints:${NC}"
        echo -e "${YELLOW}   🌐 Main API: http://localhost:${PORT}${NC}"
        echo -e "${YELLOW}   📚 API Docs: http://localhost:${PORT}/docs${NC}"
        echo -e "${YELLOW}   🏥 Health Check: http://localhost:${PORT}/health${NC}"
        echo -e "${YELLOW}   📈 Monitoring: http://localhost:${PORT}/monitoring/dashboard${NC}"
        echo -e "${YELLOW}   👤 Admin: http://localhost:${PORT}/admin/monitoring-summary${NC}"
    else
        echo -e "${RED}❌ Failed to start container!${NC}"
        exit 1
    fi
}

# Function to show container status
show_status() {
    echo -e "${BLUE}📊 Container Status:${NC}"
    podman ps -a --filter name=${CONTAINER_NAME}
    
    echo -e "\n${BLUE}📋 Container Logs (last 10 lines):${NC}"
    podman logs --tail 10 ${CONTAINER_NAME}
}

# Function to show help
show_help() {
    echo -e "${BLUE}🔧 Available Commands:${NC}"
    echo -e "${YELLOW}   ./build_and_run.sh build    ${NC}- Build container image"
    echo -e "${YELLOW}   ./build_and_run.sh run      ${NC}- Run container"
    echo -e "${YELLOW}   ./build_and_run.sh restart  ${NC}- Stop, rebuild, and run"
    echo -e "${YELLOW}   ./build_and_run.sh stop     ${NC}- Stop container"
    echo -e "${YELLOW}   ./build_and_run.sh logs     ${NC}- Show container logs"
    echo -e "${YELLOW}   ./build_and_run.sh status   ${NC}- Show container status"
    echo -e "${YELLOW}   ./build_and_run.sh shell    ${NC}- Access container shell"
    echo -e "${YELLOW}   ./build_and_run.sh clean    ${NC}- Remove container and image"
    echo -e "${YELLOW}   ./build_and_run.sh help     ${NC}- Show this help"
}

# Main script logic
case "$1" in
    "build")
        check_podman
        build_image
        ;;
    "run")
        check_podman
        cleanup_container
        run_container
        show_status
        ;;
    "restart")
        check_podman
        cleanup_container
        build_image
        run_container
        show_status
        ;;
    "stop")
        echo -e "${YELLOW}🛑 Stopping container...${NC}"
        podman stop ${CONTAINER_NAME}
        echo -e "${GREEN}✅ Container stopped${NC}"
        ;;
    "logs")
        echo -e "${BLUE}📋 Container Logs:${NC}"
        podman logs -f ${CONTAINER_NAME}
        ;;
    "status")
        show_status
        ;;
    "shell")
        echo -e "${BLUE}🐚 Accessing container shell...${NC}"
        podman exec -it ${CONTAINER_NAME} /bin/bash
        ;;
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up everything...${NC}"
        podman stop ${CONTAINER_NAME} 2>/dev/null
        podman rm ${CONTAINER_NAME} 2>/dev/null
        podman rmi ${IMAGE_NAME}:${IMAGE_TAG} 2>/dev/null
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    "help")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Invalid command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac