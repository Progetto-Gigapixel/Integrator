# Modulo Find Color Checker
Questo modulo software si occupa dell’identificazione, manipolazione e analisi di quadrilateri e di una particolare struttura chiamata **Color Checker** in immagini digitali. Il Color Checker è una tavola di riferimento composta da una griglia di colori nota, comunemente utilizzata in fotografia e imaging per la calibrazione cromatica.

### Funzioni principali del modulo

1. **Gestione dei quadrilateri**
   - Diverse funzioni si occupano di riordinare i vertici di un quadrilatero (`reorder_quadrilateral`), verificare la coerenza del loro ordine (`is_consistent_order`), determinare l’orientamento (`is_cropping_quadrilateral_horizontal`), e calcolare la matrice omografica (`compute_homography_matrix`) per trasformazioni geometriche (es. prospettiche).
   - Funzioni come `get_original_coordinates`, `recompute_resampled_coordinates`, e `recover_relative_patch_coordinates` consentono di manipolare e recuperare coordinate di punti, utili per adattare il quadrilatero a nuove prospettive o risoluzioni.

2. **Analisi colori e patch**
   - Funzioni come `retrieve_rgb_from_image` estraggono valori RGB dalle immagini in determinate posizioni (tipicamente corrispondenti ai riquadri della Color Checker).
   - `find_white_index` serve probabilmente a individuare il riquadro bianco sulla Color Checker, importante come riferimento di calibrazione.

3. **Calcolo dell’angolo di rotazione**
   - `get_rotation_angle`: determina l’angolazione del quadrilatero rispetto all’asse orizzontale, utile per correggere l’orientamento dell’immagine.

---

### Classe principale: `FindColorCheckerSegmentation`

Questa classe rappresenta il cuore del modulo e ha il compito di individuare la Color Checker all’interno di un’immagine.

- **Attributi**
  - `settings`: parametri di configurazione dell’algoritmo di ricerca.
  - `color_checker_data`: dati o modello con la descrizione del Color Checker atteso.

- **Metodi**
  - `__init__`: inizializzazione della classe con le opportune impostazioni.
  - `find`: esegue l’algoritmo di segmentazione e rilevamento del Color Checker all’interno dell’immagine, utilizzando anche le funzioni di manipolazione geometrica e di estrazione colore.
  - `_prepare_data`: prepara i dati necessari per il riconoscimento, ad esempio normalizzando l’immagine o applicando filtri.
  - `_extract_colors_from_image`: raccoglie i dati RGB effettivi dai riquadri della Color Checker rilevata.

### Metodo Find
Il metodo `find`, appartenente alla classe incaricata della segmentazione della Color Checker, restituisce tipicamente una **rappresentazione strutturata dei dati colore** rilevati nei riquadri della Color Checker trovata nell’immagine fornita.
Questa rappresentazione è solitamente sotto forma di una **matrice** (o array, tipicamente NumPy) contenente i valori RGB (o XYZ o altro spazio colore) dei patch – cioè, le misurazioni dei singoli riquadri di colore rilevati dal modulo nell’immagine d’ingresso.
- Se la Color Checker **non viene rilevata**, `find` restituisce . `None`

#### Utilizzo in `core.find_n_compare_patches`
1. **Rilevamento patch**
    - `current_measured_patches = self.color_checker_finder.find(current_image)`
    - Qui il metodo cerca di individuare la Color Checker e di estrarre i valori colore dei suoi patch dall’immagine corrente.

2. **Gestione del caso di errore**
    - Se `find` restituisce (ossia la Color Checker non viene rilevata), il metodo gestisce la situazione a seconda del contesto (ad es., solleva un’eccezione in ambiente GUI o logga un messaggio in console). `None`

3. **Comparazione dei patch**
    - Se la Color Checker è stata trovata e i valori dei patch sono disponibili, questi vengono passati a uno strumento di comparazione (`self.comparer.run(current_measured_patches, current_step)`).
    - Generalmente, questa comparazione prevede il confronto dei valori colore misurati con i valori di riferimento attesi (ad esempio usando un indice di differenza colore, come DeltaE 2000).

4. **Restituzione dei risultati**
    - Il metodo restituisce una **coppia di valori**:
        - `current_de00`: il risultato della comparazione (tipicamente una misura numerica che rappresenta la fedeltà cromatica, ad esempio l’errore di colore totale o per patch).
        - `current_measured_patches`: l’insieme dei valori colore misurati nei patch trovati della Color Checker.

---

### In sintesi

Questo modulo implementa un sistema che, dato un’immagine, è in grado di:
- **Riconoscere** la presenza e posizione della Color Checker;
- **Correggere l’orientamento** e l’aspetto geometrico della carta di riferimento anche se inclinata o deformata prospetticamente;
- **Estrarre i valori colore** dalle varie aree;
- **Fornire supporto** ad operazioni di calibrazione automatica o verifica della fedeltà cromatica nell’ambito di flussi di lavoro di imaging scientifico, industriale o fotografico.

L’insieme delle funzioni e della classe permette quindi una gestione robusta e automatizzata delle tavole Color Checker nelle immagini digitali.