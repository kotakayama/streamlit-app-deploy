# streamlit-app-deploy

## OCR Notes (Tesseract)

This app includes an OCR fallback for scanned/bitmap PDFs. It requires:

- The Python package `pytesseract` (listed in `requirements.txt`).
- The Tesseract binary installed on the host system.

Installation hints:

- macOS (Homebrew): `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows: Download installer from https://github.com/tesseract-ocr/tesseract or use Chocolatey: `choco install tesseract`.

After installing the binary, restart the app environment so `pytesseract` can find the executable. If Tesseract is not installed the app will show a friendly warning and ask the user to install it.