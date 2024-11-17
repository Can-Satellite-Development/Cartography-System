# Schritt 1: Verwende ein Basis-Image, das Python enthält
FROM python:3.11.7

# Schritt 2: Arbeitsverzeichnis im Container erstellen
WORKDIR /app

# Schritt 3: Die requirements.txt in das Arbeitsverzeichnis kopieren
COPY requirements.txt .

# Schritt 4: Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# Schritt 5: Den Rest des Codes in den Container kopieren
COPY . .

# Schritt 7: Den Startbefehl definieren
CMD ["python", "dashboard.py"]
