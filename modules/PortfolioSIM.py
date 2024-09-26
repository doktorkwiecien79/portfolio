import pandas as pd
import numpy as np

class PortfolioSimulator:
    def __init__(self, pliki_cenowe):
        """
        Inicjalizuje PortfolioSimulator z listą plików CSV zawierających dane cenowe.

        Parametry:
        - pliki_cenowe: lista ścieżek do plików CSV. Każdy plik CSV powinien mieć kolumny 'Date' i 'Close'.
        """
        self.pliki_cenowe = pliki_cenowe
        self.ceny = self._wczytaj_ceny()

    def _wczytaj_ceny(self):
        """
        Wczytuje dane cenowe z plików CSV i łączy je w jeden DataFrame.

        Zwraca:
        - DataFrame z datami jako indeks i cenami zamknięcia aktywów jako kolumny.
        """
        dane_cenowe = {}
        for plik in self.pliki_cenowe:
            df = pd.read_csv(plik, index_col='Date', parse_dates=True)
            symbol = plik.split('.')[0]  # Zakłada, że nazwa pliku to 'SYMBOL.csv'
            dane_cenowe[symbol] = df['Close']
        polaczone_ceny = pd.DataFrame(dane_cenowe)
        return polaczone_ceny

    def oblicz_stopy_zwrotu(self):
        """
        Oblicza dzienne stopy zwrotu dla każdego aktywa w portfelu.

        Zwraca:
        - DataFrame dziennych stóp zwrotu.
        """
        stopy_zwrotu = self.ceny.pct_change().dropna()
        return stopy_zwrotu

    def oblicz_statystyki_portfela(self, wagi, stopa_bezryzyka=0.0):
        """
        Oblicza statystyki portfela na podstawie wag aktywów.

        Parametry:
        - wagi: słownik, gdzie klucze to symbole aktywów, a wartości to wagi (muszą sumować się do 1)
        - stopa_bezryzyka: stopa zwrotu wolna od ryzyka (domyślnie 0.0)

        Zwraca:
        - Słownik zawierający oczekiwaną stopę zwrotu, zmienność i współczynnik Sharpe'a portfela.
        """
        stopy_zwrotu = self.oblicz_stopy_zwrotu()
        srednie_zwroty = stopy_zwrotu.mean()
        macierz_kowariancji = stopy_zwrotu.cov()

        # Upewnij się, że wagi są w odpowiedniej kolejności
        symbole = self.ceny.columns.tolist()
        w = np.array([wagi[symbol] for symbol in symbole])

        # Oblicz oczekiwaną stopę zwrotu portfela
        zwrot_portfela = np.dot(w, srednie_zwroty)

        # Oblicz zmienność portfela
        zmiennosc_portfela = np.sqrt(np.dot(w.T, np.dot(macierz_kowariancji, w)))

        # Oblicz współczynnik Sharpe'a
        sharpe_ratio = (zwrot_portfela - stopa_bezryzyka) / zmiennosc_portfela

        statystyki = {
            'Oczekiwana Stopa Zwrotu': zwrot_portfela,
            'Zmienność': zmiennosc_portfela,
            'Współczynnik Sharpe': sharpe_ratio
        }
        return statystyki
