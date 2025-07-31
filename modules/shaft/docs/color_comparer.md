# Patch Comparer

il PatchComparer è uno strumento per confrontare le patch (aree) del Color Checker.

1. **Scopo Principale**:
Il PatchComparer serve per confrontare i valori RGB tra le patch di riferimento e quelle misurate nel Color Checker.

2. **Struttura Base**:
```python
class PatchComparer:
    def __init__(self, reference_values):
        """
        :param reference_values: lista di valori RGB di riferimento per ogni patch
        """
        self.reference_values = np.array(reference_values)
```


3. **Funzionalità Principali**:

   a) **Calcolo dell'Errore**:
```python
def calculate_error(self, measured_values, error_metric='rmse'):
       """
       Calcola l'errore tra i valori misurati e quelli di riferimento
       
       :param measured_values: valori RGB misurati
       :param error_metric: metrica di errore ('rmse', 'mae', ecc.)
       :return: valore dell'errore
       """
```

   - RMSE (Root Mean Square Error): radice quadrata della media degli errori al quadrato
   - MAE (Mean Absolute Error): media degli errori assoluti

   b) **Normalizzazione dei Valori**:
```python
def normalize_values(self, values):
       """
       Normalizza i valori RGB nell'intervallo [0,1]
       """
```


4. **Utilizzo Tipico**:
```python
# Esempio di utilizzo
reference_values = [...] # Valori RGB di riferimento
comparer = PatchComparer(reference_values)

# Confronto con valori misurati
measured_values = [...] # Valori RGB misurati
error = comparer.calculate_error(measured_values)
```


5. **Caratteristiche Importanti**:
   - Gestisce confronti multi-dimensionali (RGB)
   - Supporta diverse metriche di errore
   - Normalizza automaticamente i valori se necessario
   - Può gestire sia valori singoli che batch di misurazioni

6. **Processo di Confronto**:
   1. Acquisizione dei valori di riferimento
   2. Normalizzazione dei valori (se necessario)
   3. Confronto con i valori misurati
   4. Calcolo dell'errore usando la metrica specificata
   5. Restituzione del risultato

7. **Metriche di Errore Comuni**:
```python
def rmse_error(self, measured, reference):
       """
       Calcola l'errore RMSE
       """
       return np.sqrt(np.mean((measured - reference) ** 2))

   def mae_error(self, measured, reference):
       """
       Calcola l'errore MAE
       """
       return np.mean(np.abs(measured - reference))
```


8. **Vantaggi**:
   - Flessibilità nella scelta delle metriche
   - Facilità di estensione per nuove metriche
   - Gestione robusta degli errori
   - Possibilità di confronti batch

9. **Esempio Pratico**:
```python
# Creazione del comparer
reference_patches = [
    [1.0, 0.0, 0.0],  # Rosso
    [0.0, 1.0, 0.0],  # Verde
    [0.0, 0.0, 1.0]   # Blu
]
comparer = PatchComparer(reference_patches)

# Misurazione
measured_patches = [
    [0.95, 0.05, 0.05],  # Rosso misurato
    [0.05, 0.92, 0.03],  # Verde misurato
    [0.02, 0.03, 0.94]   # Blu misurato
]

# Calcolo errore
error = comparer.calculate_error(measured_patches)
print(f"Errore di misurazione: {error}")
```


10. **Best Practices**:
    - Sempre normalizzare i valori prima del confronto
    - Utilizzare la metrica più appropriata per il caso d'uso
    - Controllare la corrispondenza delle dimensioni
    - Gestire i casi limite (valori nulli, dimensioni errate)

Il PatchComparer è quindi uno strumento fondamentale per:
- Valutare la qualità della calibrazione del colore
- Verificare l'accuratezza delle misurazioni
- Identificare problemi nella riproduzione del colore
- Fornire feedback quantitativo sulla qualità dell'immagine

