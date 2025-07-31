# Note Analisi Codebase

## Scopo
Parallelizzare analisi del color checker
- **Fase 0**: Ritagliare color checker e creare nuova immagine con arlecchino di patches per elaborare solo l'immagine ridotta.
### Fase 0
#### decode_raw.py
Decode raw presenta gia' elementi di parallelizzazione, ma non vengono sfruttati. Il metodo decode raw restituisce una immagine.
#### find_color_checker 
Identifica il color_checker e restituisce una struttra dati **JSON** con le dimensioni di ogni patch e il colore al suo interno.

### Ipotesi
Usare il find_color_checker per trovare le patches e ritagliare l'immagine. Dopo il decode_raw o dentro il decode_raw? 
