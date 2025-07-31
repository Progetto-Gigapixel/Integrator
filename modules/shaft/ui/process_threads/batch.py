import time  # For simulating processing delay

from PyQt5.QtCore import QThread, pyqtSignal


class BatchProcessThread(QThread):
    progressUpdated = pyqtSignal(int)  # Signal for progress updates
    processingComplete = pyqtSignal()  # Signal for processing completion

    def __init__(
        self, mainFolder, ccNamingConvention, destFolder, outputFormat, colorSpace
    ):
        super().__init__()
        self.mainFolder = mainFolder
        self.ccNamingConvention = ccNamingConvention
        self.destFolder = destFolder
        self.outputFormat = outputFormat
        self.colorSpace = colorSpace

    def run(self):
        # Simulated processing logic
        for i in range(101):  # Simulate 0 to 100%
            self.progressUpdated.emit(i)  # Update progress
            time.sleep(0.1)  # Simulate processing delay

        self.processingComplete.emit()  # Notify processing complete
