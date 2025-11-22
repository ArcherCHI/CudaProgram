#!/bin/bash

# SSH connection script to remote CUDA server
# Usage: ./connect.sh

echo "Connecting to remote CUDA server..."
echo "Host: cc@10.191.131.213"
echo ""

ssh -i cuda.pem -o StrictHostKeyChecking=no cc@10.191.131.213
