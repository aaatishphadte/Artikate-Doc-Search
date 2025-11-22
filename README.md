Activate Environment using venv

## Requirements Installation use UV or PIP package Manage
pip install -r requirements.txt

## Ingest Files to model  [Place ingested Docs file in docs Folder]
python -m backend.knowledge_base.ingestion

## VectorDB MIGRATION
python .\main.py makemigrations
python .\main.py migrate

## Run the application
python .\main.py runserver
