import multiprocessing

from log.logger import logger


class AnalysisProcess:
    def __init__(self, core_instance):
        self.core_instance = core_instance
        self.process = None
        self._pause_event = multiprocessing.Event()  # Evento per gestire la pausa
        self._pause_event.set()  # Inizia in stato "non in pausa"
        self._stop_event = multiprocessing.Event()  # Evento per fermare il processo

    def _run_analysis(self):
        """Esegue il core_instance.analysis_mode in un processo separato"""
        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()  # Attende fino a quando non viene ripreso
                self.core_instance.analysis_mode()  # Esegui l'analisi
                break
        except Exception as e:
            logger.error(f"Errore durante l'analisi nel processo separato: {e}")

    def start(self):
        """Avvia il processo di analisi"""
        self.process = multiprocessing.Process(target=self._run_analysis)
        self.process.start()

    def pause(self):
        """Metti in pausa il processo"""
        self._pause_event.clear()  # Imposta lo stato di pausa

    def resume(self):
        """Riprendi il processo"""
        self._pause_event.set()  # Rimuovi la pausa

    def stop(self):
        """Ferma il processo"""
        self._stop_event.set()  # Imposta l'evento di stop
        if self.process and self.process.is_alive():
            self.process.terminate()  # Termina il processo
            self.process.join()  # Assicurati che il processo venga chiuso
