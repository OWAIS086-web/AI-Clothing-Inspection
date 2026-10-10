#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 -f CERT_FILE"
    echo "  -f CERT_FILE  Path to the certificate file to install"
}

# Parse command-line options
while getopts "f:" opt; do
    case ${opt} in
        f ) CERT_FILE=$OPTARG ;;
        * ) print_usage
            exit 1 ;;
    esac
done

# Check if CERT_FILE is set
if [ -z "$CERT_FILE" ]; then
    echo "Error: Certificate file not specified."
    print_usage
    exit 1
fi

# Check if CERT_FILE exists
if [ ! -f "$CERT_FILE" ]; then
    echo "Error: Certificate file '$CERT_FILE' does not exist."
    exit 1
fi

# Define destination directory
DEST_DIR="/usr/local/share/ca-certificates/"

# Copy the certificate file
echo "Copying $CERT_FILE to $DEST_DIR"
sudo cp "$CERT_FILE" "$DEST_DIR"

# Update CA certificates
echo "Updating CA certificates..."
sudo update-ca-certificates

echo "Certificate installed and CA certificates updated successfully."
