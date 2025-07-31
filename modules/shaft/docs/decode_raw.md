# Modulo Decode Raw
Il modulo illustrato implementa una classe Python denominata `RawDecoder` che si occupa della decodifica di file immagine RAW, comunemente utilizzati nelle fotocamere digitali per salvare dati non compressi dai sensori. L'obiettivo principale di questo modulo è quello di trasformare un file RAW (o un'intera directory di file RAW) in una rappresentazione numerica più facilmente lavorabile attraverso librerie come NumPy.

### Funzionalità principali

- **Gestione di diversi formati RAW**: supporta sia file RAW veri e propri che file TIFF usati come "pseudo raw" (ad esempio, esportati in ProPhoto).
- **Decodifica singolo file**: tramite `decode_raw_image`, il decoder apre un file RAW, estrae le informazioni necessarie per la corretta conversione dei dati dal sensore, ne processa la matrice colore (ColorMatrix1 se disponibile nei metadati EXIF) e riporta il risultato in spazio colore XYZ.
- **Decodifica directory**: con `decode_raw_directory`, il decoder può gestire una cartella, processando ogni file RAW in parallelo usando `ThreadPoolExecutor` per velocizzare il flusso di lavoro.
- **Estrazione della matrice colore**: la funzione `extract_colormatrix1` legge dai metadati EXIF degli scatti la matrice colore `ColorMatrix1`, fondamentale per una conversione fedele dello spazio colore.
- **Gestione degli errori**: implementa una serie di eccezioni personalizzate, logging degli errori e messaggi localizzati per migliorare la robustezza dell’esecuzione.

### Utilizzo tipico

1. **Inizializzazione**: si crea un’istanza di `RawDecoder` specificando l’equipaggiamento, il percorso di input (file o cartella), il formato di output e la modalità operativa.
2. **Chiamata della decodifica**: l’utente invoca `decode_raw` per elaborare il file o la directory.
3. **Risultato**: il metodo restituisce un array NumPy con i dati immagine decodificati (tipicamente in spazio XYZ), che può essere utilizzato per analisi successive o conversioni in altri formati.

### Nota sul termine `result_img`

Nel contesto del modulo, `result_img` è la variabile che raccoglie il risultato del processo di decodifica, ovvero l’immagine convertita in un array numerico, pronto per essere utilizzato, visualizzato o salvato.

---

**In sintesi:**  
Questo modulo software automatizza la conversione di file RAW in dati numerici manipolabili attraverso NumPy, integrando gestione di metadati, conversione colore avanzata e parallelizzazione delle operazioni su più file. Si rivolge a flussi di lavoro fotografici professionali e scientifici che richiedono l’accesso diretto ai dati dal sensore della fotocamera.