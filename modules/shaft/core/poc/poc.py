import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import bm3d
import cv2
import imageio
import lensfunpy
import numpy as np
import rawpy
from scipy.optimize import least_squares

from utils.utils import read_raw_image


class Attrezzatura:
    def __init__(self, file_raw_path):
        self.metadati = self.estrai_exif_da_raw(file_raw_path)

        self.marca_fotocamera = self.metadati.get("Make")
        self.modello_fotocamere = self.metadati.get("Model")
        self.apertura_focale = self.metadati.get("FocalLength")  # in millimetri
        self.apertura = self.metadati.get("FNumber")
        self.distanza_focale = 1.0  # valore di default in metri

        self.velocita_otturatore = self.metadati.get("ExposureTime")
        self.iso = self.metadati.get("ISO")
        self.temperatura_colore, self.tinta = self.estrai_info_bilanciamento_bianco(
            file_raw_path
        )

        # Proprietà per le correzioni
        self.parametri_correzione_distorsione = None
        self.parametri_correzione_vignettatura = None
        self.coefficiente_trasformazione_M = None
        self.coefficiente_trasformazione_E = None
        self.matrice_correzione_luminanza = None
        self.parametri_denoising = None
        self.matrice_CCM = None
        self.coefficienti_polynomiali = None

        print("\nAttrezzatura inizializzata con successo")
        print(f"Marca fotocamera: {self.marca_fotocamera}")
        print(f"Modello fotocamera: {self.modello_fotocamere}")
        print(f"Apertura focale: {self.apertura_focale} mm")
        print(f"Apertura: f/{self.apertura}")
        print(f"Distanza focale: {self.distanza_focale} m")
        print(f"Velocità otturatore: {self.velocita_otturatore} s")
        print(f"ISO: {self.iso}")
        print(f"Temperatura colore: {self.temperatura_colore} K")
        print(f"Tinta: {self.tinta}")

    @staticmethod
    def estrai_exif_da_raw(file_raw_path):
        print(f"\nEstrazione dei metadati da: {file_raw_path}")
        # Assicurati che il percorso di ExifTool sia corretto per il tuo sistema
        comando = ["exiftool", "-j", file_raw_path]

        # Esegue il comando e cattura l'output
        try:
            risultato = subprocess.run(
                comando, capture_output=True, text=True, check=True
            )
            metadati = json.loads(risultato.stdout)[
                0
            ]  # Converte l'output JSON in un dizionario Python
            print("Metadati estratti con successo")
            return metadati
        except subprocess.CalledProcessError as e:
            print(f"Errore durante l'estrazione dei metadati: {e}")

    @staticmethod
    def estrai_info_bilanciamento_bianco(file_raw_path):
        with rawpy.imread(file_raw_path) as raw:
            # Accesso ai dati del bilanciamento del bianco automatico (AWB)
            temperatura_colore, tinta = (
                raw.camera_whitebalance[0],
                raw.camera_whitebalance[1],
            )
            # In alternativa, per alcuni modelli di fotocamera, potrebbe essere necessario
            # accedere direttamente ai dati grezzi per ottenere la temperatura di colore
            # Questo dipende dalla specifica implementazione di LibRaw per quel modello di fotocamera

        return temperatura_colore, tinta

    def carica_profili_icc(self):
        profili = {}
        # Elenco tutti i file nella directory dei profili ICC
        for file in os.listdir(self.profili_icc_directory):
            if file.lower().endswith(".icc") or file.lower().endswith(".icm"):
                # Costruisco il percorso completo al file del profilo
                percorso_completo = os.path.join(self.profili_icc_directory, file)
                # Carico il profilo ICC utilizzando la libreria Pillow (PIL)
                try:
                    profili[file] = ImageCms.getProfile(percorso_completo)
                    print(f"Profilo {file} caricato con successo.")
                except IOError:
                    print(f"Errore durante il caricamento del profilo {file}.")
        return profili

    # Metodo per applicare un profilo ICC ad un'immagine
    def applica_profilo_colore(
        self, percorso_immagine, nome_profilo_icc, spazio_colore_uscita
    ):
        try:
            # Carica l'immagine e converte in PIL Image
            immagine = Image.open(percorso_immagine)
            # Carica il profilo colore sorgente (ad esempio quello incorporato nell'immagine o uno standard)
            profilo_sorgente = ImageCms.getProfile(percorso_immagine)
            # Carica il profilo colore di destinazione dal dizionario dei profili
            profilo_destinazione = self.profili_icc.get(f"{nome_profilo_icc}.icc")

            if not profilo_destinazione:
                raise ValueError(
                    f"Il profilo {nome_profilo_icc} non è stato trovato tra i profili caricati."
                )

            # Crea il transform per convertire dal profilo sorgente al profilo destinazione
            transform = ImageCms.buildTransformFromOpenProfiles(
                profilo_sorgente, profilo_destinazione, "RGB", spazio_colore_uscita
            )
            # Applica il transform all'immagine
            immagine_trasformata = ImageCms.applyTransform(immagine, transform)

            # Salva o ritorna l'immagine trasformata
            # immagine_trasformata.save('immagine_con_nuovo_profilo.jpg')

            return immagine_trasformata
        except Exception as e:
            print(f"Errore nell'applicazione del profilo colore: {e}")


class CocoaProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.output_directory = "Assets/Processed"

        self.riferimentiTarget = "Assets/ColorChecker/RiferimentiTarget.json"

        self.autoPatchPath = "Assets/ColorChecker/colorCheckerClassicPatchesAuto.json"
        self.manualPatchPath = (
            "Assets/ColorChecker/colorCheckerClassicPatchesManual.json"
        )

        self.resultCorrezioneGeometricaPath = "Assets/CorrezioneGeometrica/post1.png"

        # self.preCorrezioneVignettingImg = self.resultCorrezioneGeometricaPath
        # self.postCorrezioneVignettingImg = "Assets/CorrezioneVignetting/post1.png"

        self.resultBilanciamentoBiancoPath = "Assets/BilanciamentoBianco/post1.png"

        self.resultCorrezioneEsposizionePath = "Assets/CorrezioneEsposizione/post1.png"

        self.whiteBalancePath = "Assets/WhiteBalance/whiteBalance.png"

        self.postFlatFieldingImg = "Assets/FlatFielding/post1.png"

        self.resultDenoisingPath = "Assets/Denoising/post1.png"

        self.resultCCMPath = "Assets/CCM/post1.png"

        with open(self.riferimentiTarget, "r") as file:
            campioni_colore = json.load(file)["CONFIG"]["TARGET"]["referenceSRGBValues"]

        #
        self.target_white = np.array(
            campioni_colore["22"]
        )  # I valori sono già normalizzati

    # Metodo per elaborare un'immagine RAW
    def decode_raw_image(self, raw_file):
        try:
            print(f"\nElaborazione del file RAW: {raw_file}")
            # Leggi il file RAW e convertilo in un'immagine RGB
            with rawpy.imread(str(raw_file)) as raw:
                rgb_image = raw.postprocess()
                # Converti in un array numpy e ridimensiona l'immagine
                rgb_array = np.array(rgb_image)

                # Ridimensiona l'immagine utilizzando l'interpolazione bilineare
                resize_factor = 8
                new_width = rgb_array.shape[1] // resize_factor
                new_height = rgb_array.shape[0] // resize_factor
                rgb_array = cv2.resize(
                    rgb_array, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

                # Salva l'immagine elaborata come PNG
                output_file = os.path.join(
                    self.output_directory,
                    os.path.basename(raw_file).split(".")[0] + ".png",
                )
                imageio.imwrite(output_file, rgb_array)
                print(f"File elaborato con successo: {output_file}")
                return output_file
        except Exception as e:
            print(f"Errore nell'elaborazione del file {raw_file}: {e}")
            return None

    # Metodo per elaborare tutti i file RAW in una directory
    def decode_raw_directory(self):
        print(f"Inizia l'elaborazione della directory: {self.directory}")
        # Trova tutti i file RAW nella directory
        raw_files = (
            list(Path(self.directory).rglob("*.raw"))
            + list(Path(self.directory).rglob("*.nef"))
            + list(Path(self.directory).rglob("*.cr2"))
            + list(Path(self.directory).rglob("*.RAF"))
            + list(Path(self.directory).rglob("*.fff"))
        )

        # Elabora ogni file RAW
        with ThreadPoolExecutor() as executor:
            # Mappa ogni file RAW a un thread separato
            future_to_raw = {
                executor.submit(self.decode_raw_image, raw_file): raw_file
                for raw_file in raw_files
            }

            for future in as_completed(future_to_raw):
                raw_file = future_to_raw[future]
                try:
                    # Ottieni il risultato dell'elaborazione
                    result = future.result()
                except Exception as exc:
                    print(
                        f"Si è verificato un errore durante l'elaborazione del file {raw_file}: {exc}"
                    )

        with open(self.autoPatchPath, "r") as file:
            colori_target = json.load(file)

        return colori_target[21][
            "detected_color"
        ]  # Patch 22 è il bianco di riferimento

    # Metodo per trovare le patch del Color Checker nell'immagine
    def find_color_checker_patches(self, immagine_path):
        # Carica l'immagine
        img = read_raw_image(immagine_path)
        # Converti l'immagine in scala di grigi e applica un filtro di sfocatura Gaussiana
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Applica l'algoritmo di Canny per la rilevazione dei bordi
        edges = cv2.Canny(blurred, 50, 150)

        # Trova i contorni nell'immagine
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Assumi che il contorno più grande sia il Color Checker
        largest_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        with open(self.riferimentiTarget, "r") as file:
            campioni_colore = json.load(file)["CONFIG"]["TARGET"]["referenceSRGBValues"]

        center_positions = []
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            rows, cols = 4, 6
            patch_width, patch_height = w // cols, h // rows

            for row in range(rows):
                for col in range(cols):
                    patch_x = x + col * patch_width + patch_width // 2
                    patch_y = y + row * patch_height + patch_height // 2
                    color = img[patch_y, patch_x]

                    # Normalizzazione dei colori rilevati e di riferimento

                    # Nota: OpenCV memorizza i colori in formato BGR
                    center_positions.append(
                        {
                            "x": patch_x,
                            "y": patch_y,
                            "detected_color": {
                                "B": int(color[0]),
                                "G": int(color[1]),
                                "R": int(color[2]),
                            },
                        }
                    )
                    cv2.circle(img, (patch_x, patch_y), 5, (0, 255, 0), -1)

            # Salva la posizione centrale delle patch nel file JSON solo dopo aver raccolto tutti i dati
            with open(self.autoPatchPath, "w") as f:
                json.dump(center_positions, f, indent=4)

        print("Rilevazione delle patch completata")

    # Metodo per applicare la correzione geometrica all'immagine
    def correzione_geometrica(self, attrezzatura, immagine_path):
        # Carica il database Lensfunpy
        db = lensfunpy.Database()

        # Trova la fotocamera nel database
        cameras = db.find_cameras(
            attrezzatura.marca_fotocamera, attrezzatura.modello_fotocamere
        )
        if not cameras:
            return "Nessuna fotocamera trovata nel database."

        camera = cameras[0]

        # Trova l'obiettivo nel database
        lenses = db.find_lenses(camera)
        if not lenses:
            return "Nessuna lente trovata nel database."

        lens = lenses[0]

        # Carica l'immagine
        immagine = cv2.imread(immagine_path)
        if immagine is None:
            return "Impossibile caricare l'immagine."

        height, width = immagine.shape[:2]

        # Calcola i parametri di correzione per la distorsione
        mod = lensfunpy.Modifier(lens, camera.crop_factor, width, height)
        mod.initialize(
            attrezzatura.apertura_focale,
            attrezzatura.apertura,
            attrezzatura.distanza_focale,
        )

        # Ottenere i coefficienti di trasformazione per la correzione geometrica
        undist_coords = mod.apply_geometry_distortion()

        # Ritorna i coefficienti di trasformazione

        return undist_coords

    # Metodo per bilanciare il bianco dell'immagine
    def bilanciamento_bianco(self, attrezzatura, immagine_path):
        print(
            f"\nApplicazione del bilanciamento del bianco all'immagine: {immagine_path}"
        )
        immagine = cv2.imread(immagine_path)

        with open(self.autoPatchPath, "r") as file:
            colori_target = json.load(file)

        # Ottieni il colore bianco di source dalla patch e normalizzalo
        source_white = colori_target[21]["detected_color"]
        source_white = (
            np.array([source_white["R"], source_white["G"], source_white["B"]]) / 255.0
        )

        # Matrice di trasformazione di Bradford
        bradford_matrix = np.array(
            [
                [0.8951, 0.2664, -0.1614],
                [-0.7502, 1.7135, 0.0367],
                [0.0389, -0.0685, 1.0296],
            ]
        )

        inverse_bradford_matrix = np.linalg.inv(bradford_matrix)

        # Converti i punti bianchi sorgente e target nel dominio lineare XYZ
        source_white_xyz = np.dot(bradford_matrix, source_white)
        target_white_xyz = np.dot(bradford_matrix, self.target_white)

        # Formula per calcolare l'adattamento della temperatura del colore e della tinta
        adaptation_matrix_color_temp = np.diag(
            [attrezzatura.temperatura_colore, 1, attrezzatura.tinta]
        )

        # Calcola la matrice di adattamento cromatica e applica l'adattamento della temperatura del colore
        # adaptation_matrix = np.dot(adaptation_matrix_color_temp, np.diag(source_white_xyz / target_white_xyz))

        # # Calcola la matrice di adattamento cromatica
        adaptation_matrix = np.diag(source_white_xyz / target_white_xyz)

        # Calcola la matrice di trasformazione completa
        transformation_matrix = np.dot(
            np.dot(inverse_bradford_matrix, adaptation_matrix), bradford_matrix
        )

        # Applica la trasformazione all'immagine
        transformed_image = np.dot(
            immagine.reshape((-1, 3)), transformation_matrix.T
        ).reshape(immagine.shape)

        # Salva l'immagine trasformata
        cv2.imwrite(self.resultBilanciamentoBiancoPath, transformed_image)

        transformation_matrix_list = transformation_matrix.tolist()

        self.aggiorna_dati_rilevati_colori(
            self.resultBilanciamentoBiancoPath, self.autoPatchPath
        )
        print("Applicazione del bilanciamento del bianco completata")
        return transformation_matrix, self.resultBilanciamentoBiancoPath

    # Metodo per applicare la correzione dell'esposizione all'immagine
    def correzione_esposizione_1(self, attrezzatura, immagine_path):
        print(f"\nCorrezione dell'esposizione dell'immagine: {immagine_path}")

        # Luminanza di riferimento per il bianco
        with open(self.autoPatchPath, "r") as file:
            colori_target = json.load(file)

        # Ottieni il colore bianco di riferimento dalla patch e normalizzalo
        source_white_rgb = colori_target[21]["detected_color"]
        target_white_norm = (
            np.array(
                [source_white_rgb["R"], source_white_rgb["G"], source_white_rgb["B"]]
            )
            / 255.0
        )

        # Normalizza i valori RGB del target_white e calcola la sua luminanza
        luminanza_target = self.calcola_luminanza_4(target_white_norm)
        luminanza_riferimento = self.calcola_luminanza_4(self.target_white)

        # Calcola lo scostamento dell'esposizione
        scostamento_esposizione = luminanza_riferimento - luminanza_target

        # Carica l'immagine, applica la correzione avanzata
        immagine = (
            cv2.imread(immagine_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        )
        immagine_corretta = self.adatta_correzione_alla_scena(
            immagine, scostamento_esposizione, attrezzatura
        )
        immagine_corretta = np.clip(immagine_corretta, 0, 1) * 255.0

        # Salva l'immagine corretta
        cv2.imwrite(self.resultCorrezioneEsposizionePath, immagine_corretta)

        self.aggiorna_dati_rilevati_colori(
            self.resultCorrezioneEsposizionePath, self.autoPatchPath
        )
        print("Correzione dell'esposizione completata")
        return scostamento_esposizione, self.resultCorrezioneEsposizionePath

    def correzione_esposizione_2(self, attrezzatura, immagine_path):
        print(f"\nInizio correzione esposizione per l'immagine: {immagine_path}")

        # Leggi i valori di colore dal patch automatico
        with open(self.autoPatchPath, "r") as file:
            colori_target = json.load(file)

        # Ottieni il colore bianco di riferimento dalla patch e normalizzalo
        target_white_rgb = (
            np.array(
                [
                    colori_target[21]["detected_color"]["R"],
                    colori_target[21]["detected_color"]["G"],
                    colori_target[21]["detected_color"]["B"],
                ],
                dtype=np.float32,
            )
            / 255.0
        )

        luminanza_target = self.calcola_luminanza_4(target_white_rgb)
        luminanza_riferimento = self.calcola_luminanza_4(self.target_white)

        # Calcola lo scostamento dell'esposizione
        scostamento_esposizione = luminanza_riferimento - luminanza_target

        # Carica l'immagine
        immagine = cv2.imread(immagine_path, cv2.IMREAD_COLOR).astype(np.float32)

        # Applica la correzione dell'esposizione
        immagine_corretta = immagine * (1 + scostamento_esposizione)
        immagine_corretta = np.clip(immagine_corretta, 0, 1) * 255.0

        # Salva l'immagine corretta
        cv2.imwrite(
            self.resultCorrezioneEsposizionePath, immagine_corretta.astype(np.uint8)
        )

        print("Correzione dell'esposizione completata.")
        return scostamento_esposizione, self.resultCorrezioneEsposizionePath

    # Metodo per applicare il flat-fielding all'immagine
    def applica_flat_fielding(self, immagine_path):
        print(f"\nApplicazione del flat-fielding all'immagine: {immagine_path}")
        # Calcola la matrice di guadagno G
        G = self.calcola_matrice_guadagno()

        # Carica l'immagine da correggere
        im = cv2.imread(immagine_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Applica la correzione flat-field
        im_corretta = im * G

        # Normalizza l'immagine corretta per prevenire overflow/underflow
        im_corretta = cv2.normalize(im_corretta, None, 0, 255, cv2.NORM_MINMAX)

        # self.aggiorna_dati_rilevati_colori(self.resultBilanciamentoBiancoPath, self.autoPatchPath)
        print(f"Applicazione del flat-fielding completata")

    # Metodo per applicare il denoising all'immagine a colori
    def applica_denoising(self, immagine_path):
        print(f"\nApplicazione del denoising all'immagine a colori: {immagine_path}")

        # Stima del rumore sull'immagine intera
        sigma_est = self.stima_rumore(immagine_path)

        # Carica l'immagine a colori e la converte in YCrCb
        img = cv2.imread(immagine_path)
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # Estrai il canale Y (luminanza)
        Y, Cr, Cb = cv2.split(img_YCrCb)

        # Immagine per il risultato finale del canale Y denoised
        Y_denoised = np.zeros_like(Y)

        # Dimensioni dei blocchi
        blocco_size = 500

        altezza, larghezza = Y.shape

        # Calcola il numero totale di blocchi in base alle dimensioni dell'immagine e alla dimensione del blocco
        n_blocchi = (
            (larghezza + blocco_size - 1)
            // blocco_size
            * (altezza + blocco_size - 1)
            // blocco_size
        )
        print(f"Numero totale di blocchi da elaborare: {n_blocchi}")

        contatore_blocco = 1

        # Elaborazione a blocchi
        for y in range(0, altezza, blocco_size):
            for x in range(0, larghezza, blocco_size):
                print(f"Elaborazione del blocco {contatore_blocco}...")
                blocco = Y[
                    y : min(y + blocco_size, altezza),
                    x : min(x + blocco_size, larghezza),
                ]
                # Applica BM3D su ogni blocco utilizzando la stima del rumore
                blocco_denoised = self.applica_bm3d(blocco, sigma_est)
                Y_denoised[
                    y : min(y + blocco_size, altezza),
                    x : min(x + blocco_size, larghezza),
                ] = blocco_denoised
                contatore_blocco += 1

        # Riunisci i canali, utilizzando la luminanza denoised e i canali di crominanza originali
        img_YCrCb_denoised = cv2.merge([Y_denoised, Cr, Cb])

        # Converti l'immagine denoised ritorno in BGR per salvataggio/visualizzazione
        img_denoised = cv2.cvtColor(img_YCrCb_denoised, cv2.COLOR_YCrCb2BGR)

        # Salva l'immagine denoised
        cv2.imwrite(self.resultDenoisingPath, img_denoised)

        print(f"Applicazione del denoising completata")

        self.aggiorna_dati_rilevati_colori(self.resultDenoisingPath, self.autoPatchPath)
        return sigma_est, self.resultDenoisingPath

    # Metodo per applicare la correzione del colore tramite la Color Correction Matrix
    def applica_color_correction_matrix(self, immagine_path):
        print(
            f"\nApplicazione della Color Correction Matrix all'immagine: {immagine_path}"
        )

        # Carica i valori rilevati e di riferimento dei colori del color checker
        with open(self.autoPatchPath, "r") as file:
            dati_rilevati = json.load(file)

        with open(self.riferimentiTarget, "r") as file:
            dati_riferimento = json.load(file)["CONFIG"]["TARGET"][
                "referenceSRGBValues"
            ]

        # Preparazione dei dati per il calcolo della CCM
        colori_riferimento = np.array(
            [dati_riferimento[str(i + 1)] for i in range(len(dati_rilevati))]
        )
        colori_rilevati = (
            np.array(
                [
                    [
                        patch["detected_color"]["R"],
                        patch["detected_color"]["G"],
                        patch["detected_color"]["B"],
                    ]
                    for patch in dati_rilevati
                ]
            )
            / 255
        )

        # # Normalizzazione dei colori (da 0 a 1)
        # colori_riferimento = colori_riferimento / 255
        # colori_rilevati = colori_rilevati / 255

        # Definizione della funzione di errore per l'ottimizzazione
        def errore(ccm_flattened, colori_rilevati, colori_riferimento):
            ccm = ccm_flattened.reshape((3, 3))
            colori_corretti = np.dot(colori_rilevati, ccm.T)
            return (colori_corretti - colori_riferimento).flatten()

        # Calcolo della CCM tramite ottimizzazione dei minimi quadrati
        ccm_iniziale = np.array(
            [1, 0, 0, 0, 1, 0, 0, 0, 1]
        )  # CCM iniziale (matrice identità)
        risultato = least_squares(
            errore, ccm_iniziale, args=(colori_rilevati, colori_riferimento)
        )

        ccm_finale = risultato.x.reshape((3, 3))
        print("CCM calcolata con successo.")

        # Carica e applica la CCM all'immagine
        immagine = cv2.imread(immagine_path)
        immagine_corretta = np.dot(immagine, ccm_finale.T)

        # Clipping dei valori per mantenere i valori validi dell'immagine
        immagine_corretta = np.clip(immagine_corretta, 0, 255).astype(np.uint8)

        # Salva l'immagine corretta
        output_path = os.path.join(
            self.resultCCMPath, "corretta_" + os.path.basename(immagine_path)
        )
        cv2.imwrite(output_path, immagine_corretta)

        print(
            f"Applicazione della Color Correction Matrix completata. Immagine salvata in: {output_path}"
        )
        return ccm_finale, output_path

    # Metodo del cliente per calcolare la differenza di colore CIEDE2000 tra un campione e uno standard
    def deltaE2000_from_Lab(self, immagine_path, KLCH=[1, 1, 1]):
        # Caricamento dei valori LAB standard da un file JSON
        with open(self.riferimentiTarget, "r") as file:
            Labstd_dictionary = json.load(file)["CONFIG"]["TARGET"][
                "referenceLABValues"
            ]
            Labstd = np.array([value for value in Labstd_dictionary.values()])

        # Caricamento dei dati delle patch rilevate nell'immagine da un altro file JSON
        with open(self.autoPatchPath, "r") as file:
            patch_data = json.load(file)

        Labsample_list = []

        # Conversione dei valori RGB rilevati per ogni patch in valori LAB
        for patch in patch_data:
            rgb = np.uint8(
                [
                    [
                        [
                            patch["detected_color"]["B"],
                            patch["detected_color"]["G"],
                            patch["detected_color"]["R"],
                        ]
                    ]
                ]
            )
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0][0]
            Labsample_list.append(lab)

        Labsample = np.array(Labsample_list)

        # Verifica che Labstd e Labsample abbiano dimensioni compatibili
        if Labstd.shape[0] != Labsample.shape[0] or Labstd.shape[1] != 3:
            raise ValueError(
                "Le dimensioni di Labstd e Labsample devono corrispondere e essere Kx3."
            )

        # Estrazione dei componenti L, a, b dai valori LAB
        kl, kc, kh = KLCH
        Lstd, astd, bstd = Labstd[:, 0], Labstd[:, 1], Labstd[:, 2]
        Lsample, asample, bsample = Labsample[:, 0], Labsample[:, 1], Labsample[:, 2]

        # Calcolo dei componenti di cromaticità C per i colori standard e campionati
        Cstd = np.sqrt(astd**2 + bstd**2)
        Csample = np.sqrt(asample**2 + bsample**2)

        # Calcolo del fattore di correzione G per il fenomeno di adattamento cromatico non lineare
        Cabarithmean = (Cstd + Csample) / 2
        G = 0.5 * (1 - np.sqrt((Cabarithmean**7) / (Cabarithmean**7 + 25**7)))

        # Applicazione del fattore G ai componenti a per calcolare i componenti a' modificati
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample

        # Calcolo dei componenti di cromaticità C' per i colori standard e campionati dopo la correzione G
        Cpstd = np.sqrt(apstd**2 + bstd**2)
        Cpsample = np.sqrt(apsample**2 + bsample**2)

        # Calcolo degli angoli hue H in radianti per i colori standard e campionati
        hpstd = np.arctan2(bstd, apstd) % (2 * np.pi)
        hpsample = np.arctan2(bsample, apsample) % (2 * np.pi)

        # Calcolo delle differenze di luminosità (dL), cromaticità (dC), e hue (dH)
        dL = Lsample - Lstd
        dC = Cpsample - Cpstd
        dhp = (hpsample - hpstd) % (2 * np.pi)
        dhp[dhp > np.pi] -= 2 * np.pi
        dhp[Cpstd * Cpsample == 0] = 0
        dH = 2 * np.sqrt(Cpstd * Cpsample) * np.sin(dhp / 2)

        # Calcolo di parametri aggiuntivi per il modello Delta E 2000
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp -= (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp += (hp < 0) * 2 * np.pi
        hp[Cpstd * Cpsample == 0] = (
            hpsample[Cpstd * Cpsample == 0] + hpstd[Cpstd * Cpsample == 0]
        )

        Lpm502 = (Lp - 50) ** 2
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = (
            1
            - 0.17 * np.cos(hp - np.pi / 6)
            + 0.24 * np.cos(2 * hp)
            + 0.32 * np.cos(3 * hp + np.pi / 30)
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        )
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            -(((180 / np.pi * hp - 275) / 25) ** 2)
        )
        Rc = 2 * np.sqrt((Cp**7) / (Cp**7 + 25**7))
        RT = -np.sin(2 * delthetarad) * Rc

        # Calcolo finale del Delta E 2000
        return np.sqrt(
            (dL / (kl * Sl)) ** 2
            + (dC / (kc * Sc)) ** 2
            + (dH / (kh * Sh)) ** 2
            + RT * (dC / (kc * Sc)) * (dH / (kh * Sh))
        )

    @staticmethod
    def stima_rumore(image_path):
        # Carica l'immagine in scala di grigi per semplificare l'analisi del rumore
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # Calcola la deviazione standard come stima approssimativa del rumore
        # In un'applicazione reale, selezionare un'area uniforme dell'immagine per questo calcolo
        deviazione_std = np.std(img)
        return deviazione_std

    @staticmethod
    def applica_bm3d(blocco, sigma_est):
        # Assicurati di convertire il blocco correttamente e applicare bm3d
        blocco = blocco.astype(np.float32) / 255.0
        risultato_bm3d = bm3d.bm3d(blocco, sigma_psd=sigma_est / 255.0)
        risultato_bm3d = (risultato_bm3d * 255).astype(np.uint8)
        return risultato_bm3d

    @staticmethod
    def calcola_matrice_guadagno():
        # Carica l'immagine flat-field (sfondo bianco)
        im_flat = cv2.imread(self.whiteBalancePath, cv2.IMREAD_GRAYSCALE).astype(
            np.float32
        )

        # Calcola il valore medio dell'immagine flat-field
        mean_flat = np.mean(im_flat)

        # Calcola la matrice di guadagno G
        G = im_flat / mean_flat

        return G

    @staticmethod
    def adatta_correzione_alla_scena(immagine, scostamento_esposizione, attrezzatura):
        # Estrazione del valore numerico dalla velocità dell'otturatore
        numeratore, denominatore = map(
            float, attrezzatura.velocita_otturatore.split("/")
        )
        tempo_otturatore = numeratore / denominatore

        # Normalizzazione dell'ISO per renderlo comparabile a una base standard (ISO 100)
        iso_factor = attrezzatura.iso / 100

        # Consideriamo l'effetto dell'apertura sulla quantità di luce che raggiunge il sensore
        # L'apertura agisce al quadrato, quindi la sua influenza sulla luminosità è quadratica, non lineare
        apertura_factor = attrezzatura.apertura**2

        # Calcoliamo il fattore di esposizione complessivo come combinazione di ISO, tempo di otturatore e apertura
        # Questo tenta di riflettere più accuratamente l'effetto combinato di questi parametri sull'esposizione finale
        esposizione_factor = iso_factor * tempo_otturatore / apertura_factor

        # Applicazione della correzione all'immagine tenendo conto della sua gamma dinamica e non linearità
        # Potremmo voler applicare la correzione in modo più sofisticato rispetto alla semplice moltiplicazione
        # Per esempio, potremmo utilizzare una funzione che modifica l'esposizione in maniera non lineare
        # per preservare i dettagli nelle alte luci e nelle ombre
        # Qui, per semplicità, continuiamo a usare un approccio lineare, ma con una base più informativa
        fattore_correzione = scostamento_esposizione * esposizione_factor

        # Assicurati che il fattore di correzione non renda l'immagine irrealisticamente luminosa o oscura
        fattore_correzione = min(max(fattore_correzione, -0.5), 0.5)

        # Applicazione della correzione all'immagine
        immagine_corretta = immagine * (1 + fattore_correzione)
        return np.clip(immagine_corretta, 0, 1)

    @staticmethod
    def calcola_luminanza_1(rgb_norm):
        # Formula approssimativa per calcolare la luminanza da RGB
        return 0.299 * rgb_norm[0] + 0.587 * rgb_norm[1] + 0.114 * rgb_norm[2]

    @staticmethod
    def calcola_luminanza_2(rgb_norm):
        gamma = 2.2  # Valore gamma comune per display standard
        # Applica la correzione gamma inversa prima del calcolo della luminanza
        rgb_gamma_corretto = [pow(c, gamma) for c in rgb_norm]
        # Formula approssimativa per calcolare la luminanza da RGB corretto
        return (
            0.299 * rgb_gamma_corretto[0]
            + 0.587 * rgb_gamma_corretto[1]
            + 0.114 * rgb_gamma_corretto[2]
        )

    @staticmethod
    def calcola_luminanza_3(rgb_norm):
        # Coefficienti per la conversione da sRGB a XYZ (assumendo sRGB)
        matrix = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        # Converti da RGB a XYZ
        xyz = np.dot(matrix, rgb_norm)
        # La componente Y dell'XYZ rappresenta la luminanza
        return xyz[1]

    @staticmethod
    def calcola_luminanza_4(rgb_norm):
        # Pesi basati sullo spazio colore sRGB
        r, g, b = rgb_norm
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @staticmethod
    def aggiorna_dati_rilevati_colori(immagine_path, file_patch_path):
        print("Aggiornamento dei dati")
        # Carica l'immagine
        img = cv2.imread(immagine_path)

        # Leggi i dati esistenti delle patch dal file JSON
        with open(file_patch_path, "r") as file:
            dati_patch = json.load(file)

        # Aggiorna i colori rilevati per ogni patch
        for patch in dati_patch:
            x, y = patch["x"], patch["y"]
            color = img[y, x]

            # Aggiorna il colore rilevato nella struttura dei dati
            patch["detected_color"]["B"] = int(color[0].item())
            patch["detected_color"]["G"] = int(color[1].item())
            patch["detected_color"]["R"] = int(color[2].item())

        # Sovrascrivi i dati nel file JSON con i dati aggiornati
        with open(file_patch_path, "w") as f:
            json.dump(dati_patch, f, indent=4)


class CocoaWorkflowManager:
    def __init__(self, directory):
        self.processor = CocoaProcessor(directory)

    def esegui_modalita_analisi(self, raw_color_checker):
        print("Eseguo analisi")

        # Inizializza l'attrezzatura con il file raw del color checker
        self.attrezzatura = Attrezzatura(raw_color_checker)

        # Decodifica l'immagine raw e restituisce il percorso dell'immagine risultante
        result_img_path = self.processor.decode_raw_image(raw_color_checker)

        # Trova le patch del color checker nell'immagine
        self.processor.find_color_checker_patches(result_img_path)
        # Stampa la media del delta E 2000 per l'immagine
        self.print_media_deltaE2000(result_img_path)

        # ! Fotocamera "Hasselblad X2D 100C" non è presente in lensfunpy
        # coefficiente = self.processor.correzione_geometrica(self.attrezzatura, result_img_path)

        # ! Fotocamera "Hasselblad X2D 100C" non è presente in lensfunpy
        # TODO: Da implementare
        # processor.correzione_vignetting()

        # Esegue il bilanciamento del bianco sull'immagine e restituisce la matrice di trasformazione lineare e il percorso dell'immagine risultante
        linear_transformation_matrix, result_img_path = (
            self.processor.bilanciamento_bianco(self.attrezzatura, result_img_path)
        )
        print(f"img: {result_img_path}")
        self.print_media_deltaE2000(result_img_path)

        # Corregge l'esposizione dell'immagine e restituisce lo scostamento dell'esposizione e il percorso dell'immagine risultante
        scostamento_esposizione, result_img_path = (
            self.processor.correzione_esposizione_1(self.attrezzatura, result_img_path)
        )
        print(f"img: {result_img_path}")
        self.print_media_deltaE2000(result_img_path)

        # ! Necessario sfondo bianco per il flat-fielding
        self.processor.applica_flat_fielding()

        # Applica il denoising all'immagine e restituisce l'estimazione del sigma e il percorso dell'immagine risultante
        sigma_est, result_img_path = self.processor.applica_denoising(result_img_path)
        print(f"img: {result_img_path}")
        self.print_media_deltaE2000(result_img_path)

        # Applica la Color Correction Matrix all'immagine e restituisce la matrice CCM e il percorso dell'immagine risultante
        ccm, result_img_path = self.processor.applica_color_correction_matrix(
            result_img_path
        )
        print(f"img: {result_img_path}")
        self.print_media_deltaE2000(result_img_path)

        print("Analisi completata")

    def print_media_deltaE2000(self, result_img_path):
        valori_deltaE = self.processor.deltaE2000_from_Lab(result_img_path)
        media_deltaE = sum(valori_deltaE) / len(valori_deltaE)
        print(f"Differenza di colore CIEDE2000: {media_deltaE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processa tutti i file RAW in una directory specificata e salva i risultati."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Il percorso della directory contenente i file RAW da elaborare.",
    )
    args = parser.parse_args()

    inputColorChecker = "Assets/Raw/Foto_0011.fff"

    manager = CocoaWorkflowManager(args.directory)

    manager.esegui_modalita_analisi(inputColorChecker)
