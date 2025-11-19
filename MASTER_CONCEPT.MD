GEMÄLDEAGENT – Master-Konzept (Markdown)
0. Executive Summary

Der Gemäldeagent ist ein intelligentes System, das automatisiert:

Gemälde / Kunstwerke erkennt

Künstler, Epoche, Stil, Material analysiert

Echtheits- und Qualitätsindikatoren bewertet

Marktwert & Preisschätzungen generiert

Provenienz-Analysen ermöglicht

Zustand, Restaurationsbedarf und Wertentwicklung einschätzt

Internet-Recherchen (Bilder, Auktionen, Marktplätze) integriert

Nutzerfreundliche Reports als PDF/DOCX erzeugt

Er kann als lokales Tool, Web-Service oder Agent in einem Multi-Agent-System laufen.

1. Zielsetzung

Der Gemäldeagent soll:

Kunstwerke anhand eines Fotos analysieren

Stil und Künstler identifizieren (ähnliche Werke, Stilrichtungen)

Echtheitsindikatoren liefern

Marktwert schätzen (Auktionspreise, Trends, Vergleichswerte)

Zustand des Bildes bewerten (Schäden, Risse, Vergilbung, Craquelé)

Empfehlungen für Restaurierung geben

Provenienz-Check per Websuche / Datenbanken unterstützen

PDF-Report für Interessenten / Versicherungen / Sammler erzeugen

Optionale tägliche Überwachung von Auktionshäusern & Marktplätzen

2. Systemarchitektur
2.1 Modularer Aufbau
/gemäldeagent
 ├── backend/
 │    ├── vision/
 │    ├── analysis/
 │    ├── valuation/
 │    ├── provenance/
 │    ├── pdf_reports/
 │    ├── web_scraper/
 ├── frontend/
 ├── agents/
 ├── data/
 └── docs/

3. Hauptkomponenten
3.1 Bildverarbeitung (Vision Layer)
Funktionen

Bild laden und normalisieren

Farbraum-Analyse

Stilklassifikation (CNN/ViT model)

Merkmalsextraktion (Brushstroke, Texture, Craquelé)

Objekt-/Motiv-Erkennung

Modelle

CLIP / SigLIP

Vision Transformer (ViT)

CNN Encoder

Werkzeug für OCR (falls Signaturen sichtbar)

3.2 Künstler- & Stil-Erkennung
Ziele

Abgleich gegen Referenzdatenbank

Ähnlichkeitsmatching anhand eines Embedding-Vektorraums

Kategorisierung nach:

Epoche (Barock, Renaissance, Impressionismus …)

Stilrichtung (Expressionismus, Kubismus …)

Künstlerstil-Signaturen (z. B. Van Gogh Brushstroke Pattern)

Technische Umsetzung

Embedding-Datenbank (FAISS/DuckDB)

Cosine Similarity Matching

Threshold-basierte Sicherheitsscoring

3.3 Echtheitsindikator-Modul
Bewertet

Signaturkonsistenz

Material & Farbpalette (Pigment-Analyse über Farbspektren)

Brushstroke-Pattern Vergleich

Craquelé-Muster

Herkunftsdokumentation (OCR + Websuche)

Vergleich mit Fälschungsmustern (GAN-Fake Detector optional)

Output

Score: 0–100

Erklärungen pro Merkmal (transparente Bewertung)

3.4 Zustand & Schadensanalyse
Erkannte Defekte

Craquelé

Vergilbung

Feuchtigkeitsschäden

Risse

Abblätterungen

Rahmenbeschädigung

Techniken

Kanten-Analyse

Texturvergleich

Heatmap-Highlighting

3.5 Marktwert-Schätzung
Datenquellen

Online-Auktionshäuser (z. B. Dorotheum, Sotheby’s, Christie’s)

Marktplätze (eBay, Willhaben)

Historische Preisdatenbank

Style-and-Artist-Similarity Weighted Pricing

Berechnung

Regressionsmodell (price prediction)

Vergleichswerk-Ähnlichkeit

Künstlertrendanalyse

Zustand & Echtheitsscore wirken preisadjustierend

Ergebnis

Minimum / Maximum / Erwartungswert

3 Confidence-Level (low / medium / high)

3.6 Provenienz-Analyse
Komponenten

Websuche nach:

Künstler

Motiv

Bildtitel (falls bekannt oder per OCR erkannt)

Auktionshistorie des Kunstwerks

Hash-Vergleich (Reverse Image Search)

Werkzeuge

Lokales LLM für Web-Ergebnis-Interpretation

Google/Bing Custom Search API (optional)

Reverse Image tools (lokal oder external)

3.7 PDF-Report Generator
Inhalt

Bild des Gemäldes

Künstler-/Stilanalyse

Echtheitsscore

Zustand & Schadensanalyse

Marktwertschätzung

Provenienz Hinweise

Empfehlung (Restaurierung, Versicherung, Verkauf)

Technologie

Reportlab / DOCX (konfigurierbar)

4. Agenten-Architektur
4.1 Hauptagent

Art: Orchestrator Agent
Funktion:

erhält Bild + optionale Metadaten

koordiniert Vision, Analyse, Bewertung und Report

generiert finalen Output

4.2 Subagenten
Agent	Aufgabe
VisionAgent	Bildanalyse, Extraktion, Stilklassifikation
ArtistMatcherAgent	Künstler- & Stilmatching, Ähnlichkeitssuche
AuthenticityAgent	Echtheitsprüfung
ConditionAgent	Schadensanalyse
ValuationAgent	Preisberechnung
ProvenanceAgent	Webrecherche, Reverse Image Search
ReportAgent	PDF/DOCX-Erstellung
5. Datenstruktur
5.1 Eingabe
{
  "image_path": "input.jpg",
  "notes": "Optional: Herkunft, Kaufpreis, Künstlervermutung"
}

5.2 Interner Standard-Datensatz
{
  "image_embeddings": [],
  "style_estimation": {
    "era": "",
    "style": "",
    "confidence": 0.92
  },
  "artist_candidates": [
    { "name": "Claude Monet", "similarity": 0.83 },
    { "name": "Camille Pissarro", "similarity": 0.74 }
  ],
  "authenticity_score": 78,
  "condition": {
    "craquele": true,
    "yellowing": false,
    "damage_score": 0.22
  },
  "valuation": {
    "estimated_value": 45000,
    "min": 35000,
    "max": 70000,
    "confidence": "medium"
  },
  "provenance": {
    "reverse_image_hits": 3,
    "auction_history": []
  }
}

6. Technologien & Bibliotheken
6.1 Vision / ML

PyTorch

Transformers (ViT, CLIP)

OpenCV

Scikit-image

6.2 Backend

FastAPI

Python 3.11

DuckDB (Embedding DB)

FAISS (Similarity)

6.3 Frontend

React / Next.js

Upload UI + Ergebnisansicht

6.4 Agents

LangGraph

Lokale LLM (Ollama / DeepSeek / Llama3)

6.5 Reports

Reportlab / python-docx

7. Workflow (End-to-End)

User lädt Bild hoch

VisionAgent extrahiert Features + Stil

ArtistMatcherAgent sucht ähnliche Künstler

AuthenticityAgent bewertet Echtheit

ConditionAgent erkennt Schäden

ValuationAgent schätzt Preis

ProvenanceAgent prüft Webfundstellen

ReportAgent baut PDF

Output an Benutzer

8. Erweiterungen (vNext)

Mobile App (Foto vor Ort)

AR-Vergleich mit echten Museumsdatenbanken

Live-Auktionsmonitoring

Sammler-Portfolio-Überwachung

Integration mit Versicherungssystemen

RAG-Kunstwissenbank

9. Sicherheits- & Governance-Kriterien

Lokale Bildverarbeitung (keine Cloud-Upload-Pflicht)

Wasserzeichen erkennen

Bias-Checks bei Valuations

Transparente Scoring-Erklärungen

Logging & Audit Trails für Expertenprüfungen

10. Aufgaben für Claude Code (Implementierungs-Roadmap)
P0 – Start (Minimal Viable Prototype)

 VisionPipeline implementieren

 Stilklassifikation

 Künstler-Matching mit FAISS

 Bewertungsschema erstellen

 PDF-Report erzeugen

P1 – Vollversion

 Web-Scraper

 Reverse Image Pipeline

 Schadensanalyse

 Preismodell

 Frontend-UI

P2 – Professional Edition

 Auktionsdaten-RAG

 Multi-Agenten-Version

 Portfolio-Überwachung
 
🔥 Neues Kapitel 11: Automatisches Markt-Monitoring für Kunstwerke
11.1 Zielsetzung

Das System soll automatisch:

Webseiten durchsuchen

Bilder extrahieren

erkennen, ob ein potenzielles Kunstwerk/Gemälde vorhanden ist

relevantes Objekt analysieren (Stil, Künstler, Preis, Echtheitsschätzung)

Matching gegen interne Datenbank ausführen

Alerts auslösen, wenn ein lohnendes/seltenes Werk auftaucht

Zielgruppe:

Kunstsammler

Investoren

Galerien

Private Käufer

Händler

11.2 Neue Systemkomponenten
/monitoring/
 ├── marketplace_scanner/
 ├── auction_scanner/
 ├── flea_market_scanner/
 ├── image_extractor/
 ├── deduplication/
 ├── scraping_policies/
 └── alerting/

11.3 Hauptfunktionalitäten
1. Periodisches Web-Scraping

Alle 30–120 Minuten:

HTML scraping

API scraping (falls verfügbar)

Reverse-Proxy / Browser-Steuerung (Playwright oder Selenium)

2. Bildextraktion

Automatische Extraktion aus:

Produktlisten

Einzeldetailseiten

Social Media Posts

PDFs (Auktionskataloge!)

3. Klassifikation “Ist es ein Kunstwerk?”

Vision-Modell entscheidet:

Gemälde oder nicht

Handgemalt oder Druck

Stil grob erkennen

Echtheitswahrscheinlichkeit (grobe Schätzung)

Material (Öl/Leinwand, Aquarell, Grafik, Lithografie)

4. Speicherung in Markt-Datenbank

Für jedes gefundene Objekt:

{
  "source": "ebay",
  "url": "https://....",
  "title": "Antikes Ölbild Bauernhaus",
  "price": 450,
  "currency": "EUR",
  "image_urls": ["..."],
  "analysis": {
    "style": "Impressionismus",
    "is_artwork": true,
    "artist_candidates": ["Pissarro", "Monet"],
    "value_estimate": {
      "min": 4000,
      "max": 8000
    }
  }
}

11.4 Neue Agenten
11.4.1 MarketCrawlerAgent

Der Hauptagent für Web-Scraping.

Aufgaben:

Seitenliste abarbeiten

Robots.txt respektieren

HTML laden

Produktlinks extrahieren

Regionale Filter anwenden (z.B. „Österreich“, „Deutschland“)

11.4.2 ImageDetectorAgent

Extrahiert alle Bilder und entscheidet: Ist es Kunst?

Tasks:

Bild laden

Vision pipeline anwenden

Klassifikation (Painting/Drawing/Graphic/Not relevant)

11.4.3 ArtworkAnalyzerAgent

Wenn ein Bild als "Kunst" erkannt wurde:

Stil analysieren

Künstler abgleichen

Echtheitsindikator

Preisschätzung

Resultate speichern

11.4.4 MarketValuationAgent

Aggregiert Preisdaten pro Quelle:

Durchschnittswerte

Trends

Erkennung von Schnäppchen / Fehlbewertungen

11.4.5 AlertAgent

Wenn interessante Angebote intensiv auffallen:

Kriterien:

Marktpreis < 40% des geschätzten Wertes

Ähnlichkeit zu Künstlern hoch

Ungewöhnlich rare Epoche

Seltene Motive

Starker Zustand → hoher Wert

Benachrichtigungen:

per E-Mail

per Telegram

per PDF-Report

per Web-Dashboard

11.5 Plattformliste (MVP + Erweiterung)
MVP

eBay

Willhaben

Etsy

Dorotheum

Facebook Marketplace (Screenshot-API / Mobile-API)

Erweiterungen

Christie’s

Sotheby’s

Artnet

Artprice

Kijiji

Shpock

Instagram Hashtags (“oil painting for sale”)

11.6 Scraper Technische Umsetzung
Technologie

Playwright (Browser-Automation)

BeautifulSoup4

Requests

Anti-detection Setup

Scraping Strategy

Indexseiten scannen

Links sammeln

Auf Detailseiten navigieren

Bilddaten extrahieren

Preise & Metadaten erfassen

in DB speichern

11.7 Deduplication Engine

Da viele Plattformen identische Angebote listen:

Algorithmus:

Perceptual Hash (pHash, dHash)

Duplicate by image similarity < 0.95

Duplicate by title/price heuristics

Ziel:

Keine doppelten Einträge im System.

11.8 Storage Layer Erweiterung
Neue Tabellen
market_items
id
source
url
title
description
price
currency
timestamp
image_hash
analysis_json

market_images
item_id
image_url
phash
embedding_vector

alerts
item_id
alert_level
reason
created_at

11.9 Workflow

MarketCrawlerAgent besucht Webseiten

extrahiert Links

ImageDetectorAgent filtert Nicht-Kunst

ArtworkAnalyzerAgent analysiert Kunstwerke

MarketValuationAgent bewertet Preis

Speicherungen in Datenbank

AlertAgent informiert über interessante Funde

11.10 Erweiterungen für Zukunft

Multi-Region-Scanning (EU / USA / Asien)

Deep-learning basierte Kunststil-Matching-Optimierung

Preisverlauf-Historie

Anomalieerkennung (Fake-Angebote)

Händler- und Verkäuferhistorie auswerten

Geodaten-basierte Suche („Flohmärkte in Salzburg/Bayern“)

Smartphone-App mit Foto-Matching gegen Marktplätze (Reverse Search)


11.11 Aufgaben für Claude Code (Implementation Plan)
Phase 1 – Grundstruktur

 marketplace_scanner/ erstellen

 API-Scraper für eBay & Willhaben

 Bildextraktion implementieren

 Deduplication Engine (pHash)

 Markt-Datenbank Tabellen anlegen

Phase 2 – Vision Integration

 ImageDetectorAgent an Vision-Modul binden

 Klassifikator „IsPainting“

 Embeddings + Matching einbauen

Phase 3 – Marktanalyse

 ArtworkAnalyzerAgent integrieren

 Preisvergleiche gegen interne Datenbank

 Schnäppchen-Erkennung

Phase 4 – Alert-System

 AlertAgent

 E-Mail / Telegram Integration

 Dashboard

Phase 5 – Erweiterte Plattformen

 Auktionshäuser

 Flohmarkt-Scanner

 Social Media Listener