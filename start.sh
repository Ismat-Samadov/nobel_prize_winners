#!/bin/bash
echo "Starting Active Learning NER System..."
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting server..."
cd api && python main_simple.py