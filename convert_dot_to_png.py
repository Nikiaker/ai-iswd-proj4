import os
import glob
import subprocess

def convert_dot_to_png(directory='./results/trees/dot/'):
    """
    Przeszukuje podany katalog (domyślnie bieżący) w poszukiwaniu wszystkich plików .dot
    i konwertuje je do formatu .png za pomocą narzędzia Graphviz (polecenie 'dot').
    """
    # Znajdź wszystkie pliki .dot w katalogu
    dot_files = glob.glob(os.path.join(directory, '*.dot'))

    if not dot_files:
        print("Nie znaleziono żadnych plików .dot w katalogu:", directory)
        return

    for dot_path in dot_files:
        png_path = "./results/trees/png/" + os.path.split(dot_path)[1] + '.png'
        try:
            # Wywołanie Graphviz: dot -Tpng input.dot -o output.png
            subprocess.run(
                ['dot', '-Tpng', dot_path, '-o', png_path],
                check=True
            )
            #print(f"Skonwertowano: {dot_path} → {png_path}")
        except subprocess.CalledProcessError as e:
            print(f"Błąd podczas konwersji {dot_path}: {e}")

if __name__ == "__main__":
    # Jeśli chcesz wskazać inny katalog, przekaż go jako argument funkcji:
    # convert_dot_to_png('/ścieżka/do/katalogu')
    convert_dot_to_png()
