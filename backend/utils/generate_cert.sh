#!/bin/bash

# Define variables
KEY_FILE="private.key"
CSR_FILE="request.csr"
CRT_FILE="certificate.crt"
DAYS_VALID=365

# Function to print usage
print_usage() {
    echo "Usage: $0 [-d DAYS_VALID] [-k KEY_FILE] [-c CSR_FILE] [-o CRT_FILE]"
    echo "  -d DAYS_VALID  Number of days the certificate is valid for (default: 365)"
    echo "  -k KEY_FILE    Private key file (default: private.key)"
    echo "  -c CSR_FILE    CSR file (default: request.csr)"
    echo "  -o CRT_FILE    Output CRT file (default: certificate.crt)"
}

# Parse command-line options
while getopts "d:k:c:o:" opt; do
    case ${opt} in
        d ) DAYS_VALID=$OPTARG ;;
        k ) KEY_FILE=$OPTARG ;;
        c ) CSR_FILE=$OPTARG ;;
        o ) CRT_FILE=$OPTARG ;;
        * ) print_usage
            exit 1 ;;
    esac
done

# Generate private key
echo "Generating private key..."
openssl genpkey -algorithm RSA -out ${KEY_FILE} 

# Create CSR
echo "Creating Certificate Signing Request (CSR)..."
openssl req -new -key ${KEY_FILE} -out ${CSR_FILE}

# Generate self-signed certificate
echo "Generating self-signed certificate..."
openssl x509 -req -days ${DAYS_VALID} -in ${CSR_FILE} -signkey ${KEY_FILE} -out ${CRT_FILE}

echo "Process completed."
echo "Private Key: ${KEY_FILE}"
echo "CSR: ${CSR_FILE}"
echo "Certificate: ${CRT_FILE}"
